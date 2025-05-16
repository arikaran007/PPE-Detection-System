import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from PIL import Image
import pandas as pd
import datetime
import torch

# Set confidence threshold
CONFIDENCE_THRESHOLD = 0.30

# Set page config
st.set_page_config(page_title="PPE Detection System", layout="wide")

# Initialize session state for report generation
if 'detections' not in st.session_state:
    st.session_state['detections'] = []

# Initialize session state for processed images
if 'processed_image' not in st.session_state:
    st.session_state['processed_image'] = None

def load_models():
    """Load YOLO models for PPE detection and pose estimation"""
    try:
        ppe_model = YOLO('best.pt')  # Your trained PPE model
        pose_model = YOLO('yolov8n-pose.pt')  # YOLOv8 pose model
        return ppe_model, pose_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def process_frame(frame, ppe_model, pose_model):
    """Process a single frame with both PPE and pose detection"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # PPE Detection
    ppe_results = ppe_model(frame_rgb, conf=CONFIDENCE_THRESHOLD)[0]
    
    detection_summary = []
    
    # Process PPE detections
    for box, conf, cls in zip(ppe_results.boxes.xyxy, ppe_results.boxes.conf, ppe_results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)
        confidence = float(conf)
        label = ppe_model.names[class_id]
        
        # Store detection summary
        detection_summary.append((label, confidence))
        
        # Draw bounding box
        color_map = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}
        color = color_map.get(class_id, (0, 255, 0))
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_rgb, f"{label}: {confidence:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save detections to session state for report generation
    if detection_summary:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state['detections'].extend([(timestamp, *d) for d in detection_summary])
    
    st.session_state['processed_image'] = frame_rgb
    return frame_rgb

def generate_report():
    """Generate a downloadable CSV report of detections"""
    if st.session_state['detections']:
        df = pd.DataFrame(st.session_state['detections'], columns=["Timestamp", "Detected PPE", "Confidence"])
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Report", csv, "ppe_detection_report.csv", "text/csv", key='download-csv')
    else:
        st.warning("No detections to generate a report.")

def main():
    st.title("PPE Detection System")
    st.write("Detect Personal Protective Equipment (PPE) and Body Pose in Real-Time")
    
    # Load models
    ppe_model, pose_model = load_models()
    if ppe_model is None or pose_model is None:
        st.error("Failed to load models. Please check model paths and try again.")
        return
    
    # Input selection
    input_option = st.radio("Select Input Source:", ["Image", "Video", "Webcam"])
    
    if input_option == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            processed_image = process_frame(image_np, ppe_model, pose_model)
            st.image(processed_image, caption="Processed Image", use_column_width=True)
            generate_report()
    
    elif input_option == "Video":
        uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                processed_image = process_frame(frame, ppe_model, pose_model)
                st.image(processed_image, caption="Processed Video Frame", use_column_width=True)
            
            cap.release()
            generate_report()
    
    elif input_option == "Webcam":
        run = st.checkbox('Start Webcam')
        camera = cv2.VideoCapture(0)
        
        while run:
            ret, frame = camera.read()
            if frame is not None:
                processed_image = process_frame(frame, ppe_model, pose_model)
                st.image(processed_image, caption="Processed Webcam Frame", use_column_width=True)
            else:
                st.warning("No frame available from webcam")
                break
        
        camera.release()
        generate_report()

if __name__ == '__main__':
    main()
