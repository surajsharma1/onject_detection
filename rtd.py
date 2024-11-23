import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO('C:/Users/suraj/Documents/GitHub/licence_plate_detection/runs/detect/train8/weights/best.pt')  # Update path to your trained weights

# Open webcam for real-time detection
cap = cv2.VideoCapture(0)  # '0' is usually the default webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)
    annotated_frame = results[0].plot()

    # Display the results
    cv2.imshow("Real-Time License Plate Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
