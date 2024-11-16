from argparse import ArgumentParser
import cv2
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine

# Parse arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=0,
                    help="The webcam index.")
args = parser.parse_args()

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

def run():
    # Initialize the video source from webcam or video file.
    video_src = args.cam if args.video is None else args.video
    cap = cv2.VideoCapture(video_src)
    print(f"Video source: {video_src}")

    # Get the frame size. This will be used by the following detectors.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup a face detector to detect human faces.
    face_detector = FaceDetector("assets/face_detector.onnx")

    # Setup a mark detector to detect landmarks.
    mark_detector = MarkDetector("assets/face_landmarks.onnx")

    # Setup a pose estimator to solve pose.
    pose_estimator = PoseEstimator(frame_width, frame_height)

    # Measure the performance with a tick meter.
    tm = cv2.TickMeter()

    while True:
        # Read a frame.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # If the frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Step 1: Get faces from current frame.
        faces, _ = face_detector.detect(frame, 0.7)

        if len(faces) > 0:
            tm.start()

            # Step 2: Detect landmarks.
            face = refine(faces, frame_width, frame_height, 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]

            # Run the mark detection.
            marks = mark_detector.detect([patch])[0].reshape([68, 2])

            # Convert to global image.
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Step 3: Try pose estimation.
            distraction_status, pose_vectors = pose_estimator.detect_distraction(marks)
            rotation_vector, translation_vector = pose_vectors
            
            # Check distraction
            if distraction_status:
                status_text = "Distracted"
            else:
                status_text = "Focused"
            
            cv2.putText(frame, f"Status: {status_text}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if not distraction_status else (0, 0, 255))

            tm.stop()

            # Visualize the pose
            pose_estimator.visualize(frame, pose_vectors, color=(0, 255, 0))

            # Draw axes
            pose_estimator.draw_axes(frame, pose_vectors)

        # Draw the FPS on the screen
        cv2.rectangle(frame, (0, 0), (90, 30), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"FPS: {tm.getFPS():.0f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        # Show preview
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == 27:
            break

if __name__ == '__main__':
    run()