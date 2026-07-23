#!/usr/bin/env python
# Modified from https://github.com/facebookresearch/detectron2/blob/main/tools/lazyconfig_train_net.py
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.
This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.
Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""

import glob
import json
import logging
import os

import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyCall, LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.solver import WarmupParamScheduler
from detectron2.utils import comm
from fvcore.common.param_scheduler import MultiStepParamScheduler

from customizedComponents.customizedEvalHook import (
    customEvalHook,
    customLossEval,
)
from customizedComponents.customizedTrainer import (
    customAMPTrainer,
    customSimpleTrainer,
)
from utils import get_no_label_dicts, get_rebar_dicts

logger = logging.getLogger("detectron2")


class PeriodicLastCheckpointer(hooks.HookBase):
    """
    Save a fixed-name checkpoint (model_last.pth) every `period` iterations by
    overwriting the same file. Avoids creating model_XXXXXXX.pth files that
    Google Drive moves to Trash when deleted (filling Drive quota for 30 days).
    """

    def __init__(self, checkpointer, period, file_name="model_last"):
        self.checkpointer = checkpointer
        self.period = int(period)
        self.file_name = file_name

    def after_step(self):
        if not comm.is_main_process():
            return
        next_iter = self.trainer.iter + 1
        if self.period > 0 and next_iter % self.period == 0:
            self.checkpointer.save(self.file_name)

    def after_train(self):
        if not comm.is_main_process():
            return
        # Final overwrite so last always reflects end of training.
        self.checkpointer.save(self.file_name)


def _get_segm_ap(eval_results):
    """Extract COCO mask AP from evaluator output."""
    if not eval_results:
        return None
    if "segm" in eval_results and isinstance(eval_results["segm"], dict):
        return eval_results["segm"].get("AP", None)
    # flattened keys used by some detectron2 versions / print paths
    if "segm/AP" in eval_results:
        return eval_results["segm/AP"]
    return None


def _load_best_ap(output_dir):
    path = os.path.join(output_dir, "best_metrics.json")
    if not os.path.isfile(path):
        return float("-inf")
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return float(data.get("segm_AP", float("-inf")))
    except Exception:
        return float("-inf")


def _save_best_checkpoint(cfg, checkpointer, eval_results, iteration):
    """
    Save model_best.pth when segm AP improves.
    Restores last_checkpoint so resume still points to the latest periodic ckpt.
    """
    if not comm.is_main_process():
        return
    ap = _get_segm_ap(eval_results)
    if ap is None:
        logger.warning("segm AP not found in eval results; skip best checkpoint update.")
        return

    best_ap = _load_best_ap(cfg.train.output_dir)
    if float(ap) <= best_ap:
        logger.info(
            "Current segm AP {:.4f} <= best {:.4f}; keep existing model_best.pth".format(
                float(ap), best_ap
            )
        )
        return

    last_ckpt_file = os.path.join(cfg.train.output_dir, "last_checkpoint")
    prev_last = None
    if os.path.isfile(last_ckpt_file):
        with open(last_ckpt_file, "r") as f:
            prev_last = f.read().strip()

    checkpointer.save("model_best")

    # Do not let best overwrite resume pointer.
    if prev_last:
        with open(last_ckpt_file, "w") as f:
            f.write(prev_last)

    metrics = {
        "iter": int(iteration),
        "segm_AP": float(ap),
        "bbox_AP": None,
    }
    if "bbox" in eval_results and isinstance(eval_results["bbox"], dict):
        metrics["bbox_AP"] = eval_results["bbox"].get("AP", None)

    with open(os.path.join(cfg.train.output_dir, "best_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "Updated model_best.pth at iter {} with segm AP {:.4f}".format(
            iteration, float(ap)
        )
    )


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test),
            instantiate(cfg.dataloader.evaluator),
        )
        with open(
            os.path.join(cfg.train.output_dir, "ap_steel_val.json"), "a"
        ) as f:
            json.dump(ret, f)
            f.write("\n")
        print_csv_format(ret)
        return ret


def do_source_test(cfg, model):
    if (
        "evaluator_source" in cfg.dataloader
        and "test_source" in cfg.dataloader
    ):
        ret = inference_on_dataset(
            model,
            instantiate(cfg.dataloader.test_source),
            instantiate(cfg.dataloader.evaluator_source),
        )
        with open(
            os.path.join(cfg.train.output_dir, "ap_steel_test.json"), "a"
        ) as f:
            json.dump(ret, f)
            f.write("\n")
        print_csv_format(ret)
        return ret


def do_validation_loss(cfg, model):
    if "test_source" in cfg.dataloader:
        losses = customLossEval(
            model, instantiate(cfg.dataloader.test_source), domainSource=True
        )
        with open(
            os.path.join(cfg.train.output_dir, "test_source.json"), "a"
        ) as f:
            json.dump(losses, f)
            f.write("\n")
    # if "test" in cfg.dataloader:
    #     losses = customLossEval(model, instantiate(cfg.dataloader.test), domainSource=False)
    #     with open(os.path.join(cfg.train.output_dir, "test_target.json"), "a") as f:
    #         json.dump(losses, f)
    #         f.write("\n")


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)
    train_target_loader = instantiate(cfg.dataloader.train_target)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (
        customAMPTrainer if cfg.train.amp.enabled else customSimpleTrainer
    )(model, train_loader, train_target_loader, optim)

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )

    def eval_and_save_best():
        ret = do_test(cfg, model)
        if ret is not None:
            _save_best_checkpoint(cfg, checkpointer, ret, trainer.iter)
        return ret

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            # Overwrite model_last.pth instead of rotating model_XXXXXXX.pth
            # (Drive Trash retains deleted files ~30 days and fills quota).
            PeriodicLastCheckpointer(
                checkpointer, period=cfg.train.checkpointer.period
            )
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, eval_and_save_best),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
            # customEvalHook(cfg.train.log_period * 5, lambda: do_validation_loss(cfg, model))
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    cfg.train.max_iter = 60000
    cfg.train.checkpointer.period = 200
    cfg.train.eval_period = 1000
    cfg.train.log_period = 20
    cfg.train.amp.enabled = False
    cfg.optimizer.lr = 0.001
    cfg.lr_multiplier = LazyCall(WarmupParamScheduler)(
        scheduler=LazyCall(MultiStepParamScheduler)(
            values=[1.0, 0.81, 0.73, 0.65, 0.3, 0.1, 0.01],
            milestones=[25000, 35000, 40000, 45000, 50000, 55000, 60000],
        ),
        warmup_length=0.1,
        warmup_method="linear",
        warmup_factor=0.001,
    )
    if args.num_gpus > 1:
        cfg.model.backbone.bottom_up.stem.norm = (
            cfg.model.backbone.bottom_up.stages.norm
        ) = "SyncBN"
        cfg.model.backbone.norm = "SyncBN"
    else:
        cfg.model.backbone.bottom_up.stem.norm = (
            cfg.model.backbone.bottom_up.stages.norm
        ) = "BN"
        cfg.model.backbone.norm = "BN"

    default_setup(cfg, args)
    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
