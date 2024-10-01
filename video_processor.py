import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
import os

BEHAVIORS = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 'Explosion', 
            'Fighting', 'RoadAccidents', 'Robbery', 'Shooting', 'Shoplifting', 
            'Stealing', 'Vandalism', 'NormalVideos']

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # CNN layers
        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Second block
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Fourth block
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.feature_size = 512 * (64 // 8) * (64 // 8)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, len(BEHAVIORS))
        )

    def forward(self, x):
        features = self.conv_layers(x)
        features = features.view(x.size(0), -1)
        output = self.classifier(features)
        return output

class VideoProcessor:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CustomCNN().to(self.device)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def process_frame(self, frame):
        # Chuyển frame sang PIL Image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Transform và predict
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        return predicted.item(), confidence.item()

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        
        # Lấy thông tin video
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Tạo video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        predictions = []
        confidences = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Xử lý frame
            pred, conf = self.process_frame(frame)
            predictions.append(pred)
            confidences.append(conf)
            
            # Vẽ kết quả lên frame
            behavior = BEHAVIORS[pred]
            cv2.putText(frame, f'Action: {behavior} ({conf:.2f})', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        # Kết luận cuối cùng
        final_prediction = max(set(predictions), key=predictions.count)
        avg_confidence = np.mean(confidences)
        
        return BEHAVIORS[final_prediction], avg_confidence
