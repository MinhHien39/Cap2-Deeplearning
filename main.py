import os
import shutil
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import yaml

def split_data(source_dir, train_dir, val_dir, val_split=0.2):
    print(f"Source directory: {source_dir}")
    print(f"Train directory: {train_dir}")
    print(f"Validation directory: {val_dir}")
    
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"Found {len(class_dirs)} class directories")
    
    for class_dir in class_dirs:
        print(f"Processing class: {class_dir}")
        class_source_dir = os.path.join(source_dir, class_dir)
        class_train_dir = os.path.join(train_dir, class_dir)
        class_val_dir = os.path.join(val_dir, class_dir)
        
        os.makedirs(class_train_dir, exist_ok=True)
        os.makedirs(class_val_dir, exist_ok=True)
        
        all_files = [f for f in os.listdir(class_source_dir) if os.path.isfile(os.path.join(class_source_dir, f))]
        print(f"  Found {len(all_files)} files")
        
        train_files, val_files = train_test_split(all_files, test_size=val_split, random_state=42)
        
        print(f"  Copying {len(train_files)} files to train directory")
        for file in train_files:
            shutil.copy(os.path.join(class_source_dir, file), os.path.join(class_train_dir, file))
        
        print(f"  Copying {len(val_files)} files to validation directory")
        for file in val_files:
            shutil.copy(os.path.join(class_source_dir, file), os.path.join(class_val_dir, file))
        
    print("Data split completed")

# Lấy đường dẫn tuyệt đối của thư mục hiện tại
current_dir = os.path.dirname(os.path.abspath(__file__))

# Sử dụng os.path.join để tạo đường dẫn
source_dir = os.path.join(current_dir, 'Model', 'Train')
train_dir = os.path.join(current_dir, 'dataset', 'train')
val_dir = os.path.join(current_dir, 'dataset', 'val')

# Tạo thư mục mới nếu chưa tồn tại
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

print("Starting data split...")
try:
    split_data(source_dir, train_dir, val_dir)
except Exception as e:
    print(f"An error occurred during data split: {e}")
print("Data split completed.")

print(f"Checking directory permissions:")
print(f"Source directory exists: {os.path.exists(source_dir)}")
print(f"Can read source directory: {os.access(source_dir, os.R_OK)}")
print(f"Can write to train directory: {os.access(os.path.dirname(train_dir), os.W_OK)}")
print(f"Can write to validation directory: {os.access(os.path.dirname(val_dir), os.W_OK)}")

# Tạo file data.yaml
data = {
    'train': os.path.abspath(train_dir),
    'val': os.path.abspath(val_dir),
    'nc': len(os.listdir(source_dir)),  # Số lượng lớp
    'names': sorted(os.listdir(source_dir))  # Tên các lớp
}

with open('data.yaml', 'w') as file:
    yaml.dump(data, file)

print("Creating data.yaml file...")
print("data.yaml file created.")

print("Initializing model...")
model = YOLO('yolov8n-cls.pt')
print("Model initialized.")

print("Starting training...")
try:
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=224,
        batch=32,
        name='crime_action_detection',
        verbose=True
    )
except Exception as e:
    print(f"An error occurred during training: {e}")

# Đánh giá mô hình
results = model.val()
print("Training completed.")