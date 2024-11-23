from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('C:/Users/suraj/Documents/GitHub/licence_plate_detection/runs/detect/train8/weights/best.pt')  # Update path to your best weights

# Specify the image path for inference
image_path = 'C:/Users/suraj/Documents/GitHub/licence_plate_detection/data/images/test/DALL-E-2023-05-23-10-59-19_png.rf.98edc3b9072b3fe5ee2243d23f74b773.jpg'  # Replace with your image path

# Perform inference
results = model(image_path)

for result in results:
    result.show()  # Show the image with bounding boxes around detected license plates
