import cv2
import streamlit as st
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine

def main():
    # Streamlit Title and Sidebar for inputs
    st.title("Distraction Detection App")
    video_src = st.sidebar.selectbox("Select Video Source", ("Webcam", "Video File"))
    
    # If a video file is chosen, provide file uploader
    if video_src == "Video File":
        video_file = st.sidebar.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])
        if video_file is not None:
            video_src = video_file
        else:
            st.warning("Please upload a video file.")
            return
    else:
        video_src = 0  # Webcam index
    
    # Setup the video capture and detector components
    cap = cv2.VideoCapture(video_src if video_src == 0 else video_file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_detector = FaceDetector("assets/face_detector.onnx")
    mark_detector = MarkDetector("assets/face_landmarks.onnx")
    pose_estimator = PoseEstimator(frame_width, frame_height)

    # Streamlit placeholders for images
    frame_placeholder = st.empty()
    
    while cap.isOpened():
        # Capture a frame
        frame_got, frame = cap.read()
        if not frame_got:
            break

        # Flip the frame if from webcam
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Face detection and pose estimation
        faces, _ = face_detector.detect(frame, 0.7)
        if len(faces) > 0:
            face = refine(faces, frame_width, frame_height, 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]
            marks = mark_detector.detect([patch])[0].reshape([68, 2])
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1
            
            distraction_status, pose_vectors = pose_estimator.detect_distraction(marks)
            status_text = "Distracted" if distraction_status else "Focused"
            
            # Overlay status text
            cv2.putText(frame, f"Status: {status_text}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        (0, 255, 0) if not distraction_status else (0, 0, 255))
            
            # Display the frame in Streamlit
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

if __name__ == "__main__":
    main()
