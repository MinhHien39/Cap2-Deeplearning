import torch
import torch.nn as nn
from ultralytics import YOLO
import os
import json

class CombinedModel(nn.Module):
    def __init__(self, num_classes=3):
        super(CombinedModel, self).__init__()
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CriminalDetectionSystem:
    def __init__(self, data_yaml, epochs=10, img_size=640, batch_size=16):
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.img_size = img_size
        self.batch_size = batch_size
        self.checkpoint_path = 'checkpoints'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Tạo thư mục checkpoints nếu chưa tồn tại
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
        # Khởi tạo YOLOv8
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Khởi tạo CNN
        self.cnn_model = CombinedModel().to(self.device)
        self.optimizer = torch.optim.Adam(self.cnn_model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        
        # Tải trạng thái huấn luyện trước đó nếu có
        self.start_epoch = self.load_checkpoint()

    def load_checkpoint(self):
        checkpoint_file = os.path.join(self.checkpoint_path, 'latest.pt')
        if os.path.exists(checkpoint_file):
            print("Đang tải checkpoint...")
            checkpoint = torch.load(checkpoint_file)
            self.cnn_model.load_state_dict(checkpoint['cnn_model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            return checkpoint['epoch']
        return 0

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'cnn_model_state': self.cnn_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_path, 'latest.pt'))
        print(f"Đã lưu checkpoint tại epoch {epoch}")

    def train(self):
        try:
            # Huấn luyện YOLOv8
            print("Bắt đầu huấn luyện YOLOv8...")
            self.yolo_model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                imgsz=self.img_size,
                batch=self.batch_size,
                resume=True  # Tự động tiếp tục từ checkpoint cuối cùng
            )

            # Huấn luyện CNN
            print("Bắt đầu huấn luyện CNN...")
            for epoch in range(self.start_epoch, self.epochs):
                self.cnn_model.train()
                # ... (thêm logic huấn luyện CNN ở đây)
                
                # Lưu checkpoint sau mỗi epoch
                self.save_checkpoint(epoch + 1)
                
                print(f'Epoch [{epoch+1}/{self.epochs}] completed')

        except KeyboardInterrupt:
            print("\nHuấn luyện bị dừng. Đang lưu checkpoint...")
            self.save_checkpoint(epoch + 1)
            print("Checkpoint đã được lưu. Bạn có thể tiếp tục huấn luyện sau.")

    def predict(self, image_path):
        # Thực hiện dự đoán sử dụng cả YOLOv8 và CNN
        yolo_results = self.yolo_model.predict(image_path)
        
        # Xử lý kết quả từ YOLO
        detections = []
        for result in yolo_results:
            for box in result.boxes:
                detection = {
                    'class': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                }
                detections.append(detection)
        
        return detections

# Khởi tạo và sử dụng hệ thống
def main():
    data_yaml = 'dataYolo/data.yaml'  # Đường dẫn tới file data.yml của bạn
    system = CriminalDetectionSystem(data_yaml)
    system.train()

if __name__ == "__main__":
    main()