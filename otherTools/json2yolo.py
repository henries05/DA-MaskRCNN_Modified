import os
import json
import glob
import shutil
import argparse
from pathlib import Path

def convert_labelme_to_yolo_seg(json_path, txt_path, class_mapping):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    img_w = data.get('imageWidth')
    img_h = data.get('imageHeight')
    
    if not img_w or not img_h:
        print(f"Warning: {json_path} missing image width/height. Skipping.")
        return False

    lines = []
    for shape in data.get('shapes', []):
        label = shape['label']
        if label not in class_mapping:
            continue
            
        class_id = class_mapping[label]
        points = shape['points']
        
        # Normalize points to 0-1
        normalized_points = []
        for x, y in points:
            nx = max(0.0, min(1.0, x / img_w))
            ny = max(0.0, min(1.0, y / img_h))
            normalized_points.extend([f"{nx:.6f}", f"{ny:.6f}"])
            
        if len(normalized_points) >= 6: # At least 3 points for a polygon
            lines.append(f"{class_id} " + " ".join(normalized_points))
            
    if lines:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        return True
    return False

def setup_yolo_dataset(source_dir, dest_dir, split_name, class_mapping):
    """
    Tìm tất cả các file JSON trong source_dir (bao gồm cả thư mục con).
    Ứng với mỗi JSON, tìm ảnh tương ứng và copy sang cấu trúc chuẩn YOLO.
    """
    img_dest_dir = os.path.join(dest_dir, 'images', split_name)
    lbl_dest_dir = os.path.join(dest_dir, 'labels', split_name)
    
    os.makedirs(img_dest_dir, exist_ok=True)
    os.makedirs(lbl_dest_dir, exist_ok=True)
    
    json_files = glob.glob(os.path.join(source_dir, '**', '*.json'), recursive=True)
    count = 0
    
    for json_path in json_files:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        img_filename = data.get('imagePath')
        if not img_filename:
            continue
            
        # Tìm đường dẫn thực tế của ảnh (có thể nằm cùng thư mục hoặc trong thư mục imgs/)
        json_dir = os.path.dirname(json_path)
        img_path = os.path.join(json_dir, 'imgs', os.path.basename(img_filename))
        if not os.path.exists(img_path):
            img_path = os.path.join(json_dir, os.path.basename(img_filename))
            
        if not os.path.exists(img_path):
            print(f"Không tìm thấy ảnh: {img_path}")
            continue
            
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(lbl_dest_dir, f"{base_name}.txt")
        new_img_path = os.path.join(img_dest_dir, os.path.basename(img_path))
        
        if convert_labelme_to_yolo_seg(json_path, txt_path, class_mapping):
            shutil.copy2(img_path, new_img_path)
            count += 1
            
    print(f"Đã xử lý {count} ảnh và nhãn cho tập {split_name}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert LabelMe JSONs to YOLO Segmentation format")
    parser.add_argument('--virtual-dir', type=str, default='virtualData', help="Path to virtualData")
    parser.add_argument('--real-dir', type=str, default='realData', help="Path to realData")
    parser.add_argument('--output-dir', type=str, default='yolo_dataset', help="Output directory for YOLO dataset")
    args = parser.parse_args()

    # Kiểm tra xem thư mục output đã tồn tại chưa để tránh convert lại từ đầu gây tốn thời gian
    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0:
        print(f"✅ Thư mục YOLO dataset '{args.output_dir}' đã tồn tại và có dữ liệu. Bỏ qua bước convert để tiết kiệm thời gian.")
        exit(0)

    # YOLO class mapping
    class_mapping = {
        'intersection': 0,
        'spacing': 1
    }
    
    print("🚀 Bắt đầu convert virtualData (dùng làm tập train cho YOLO)...")
    if os.path.exists(args.virtual_dir):
        setup_yolo_dataset(args.virtual_dir, args.output_dir, 'train', class_mapping)
    else:
        print(f"Thư mục {args.virtual_dir} không tồn tại!")
        
    print("🚀 Bắt đầu convert realData (dùng làm tập val cho YOLO)...")
    if os.path.exists(args.real_dir):
        setup_yolo_dataset(args.real_dir, args.output_dir, 'val', class_mapping)
    else:
        print(f"Thư mục {args.real_dir} không tồn tại!")
    
    print(f"✅ Hoàn tất! Dataset chuẩn YOLO được lưu tại: {args.output_dir}")
