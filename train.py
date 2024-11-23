from ultralytics import YOLO

# Specify the path to the configuration YAML file
data_config_path = 'C:/Users/suraj/Documents/GitHub/licence_plate_detection/data.yaml'

# Initialize YOLO model (You can use a pre-trained model or start from scratch)
model = YOLO('yolov8n.pt')  # Use 'yolov8n.pt' or another pre-trained model

# Train the model
model.train(data=data_config_path, epochs=50, imgsz=640, batch=16)

# After training, the model will be saved in runs/train/exp/
print("Training completed!")
