import cv2
import numpy as np

class PoseEstimator:
    # Existing initialization and methods...

    def rotation_matrix_to_angles(self, rotation_vector):
        """Convert rotation vector to pitch, yaw, and roll angles."""
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
        
        singular = sy < 1e-6
        if not singular:
            pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            pitch = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            yaw = np.arctan2(-rotation_matrix[2, 0], sy)
            roll = 0

        return np.degrees(pitch), np.degrees(yaw), np.degrees(roll)

    def is_distracted(self, rotation_vector):
        """Determine if the user is distracted based on head pose angles."""
        pitch, yaw, roll = self.rotation_matrix_to_angles(rotation_vector)
        
        # Define thresholds (you may need to adjust these based on testing)
        pitch_threshold = 15    # Up/Down threshold
        yaw_threshold = 20      # Left/Right threshold
        roll_threshold = 10     # Tilt threshold
        
        # Check if head is facing roughly forward
        if abs(pitch) < pitch_threshold and abs(yaw) < yaw_threshold and abs(roll) < roll_threshold:
            return False  # Focused
        else:
            return True   # Distracted

    def detect_distraction(self, points):
        """Solve pose and detect distraction status based on pose."""
        rotation_vector, translation_vector = self.solve(points)
        distraction_status = self.is_distracted(rotation_vector)
        return distraction_status, (rotation_vector, translation_vector)
