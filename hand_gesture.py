import cv2
import numpy as np
import mediapipe as mp

class HandGestureControl:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils
        self.gesture_map = {
            "thumbs_up": "Action: Volume Up",
            "thumbs_down": "Action: Volume Down",
            "fist": "Action: Pause/Play",
        }

    def detect_gesture(self, landmarks):
        """Detects gesture based on landmarks"""
        if landmarks[4][1] < landmarks[3][1]:  # Thumb rule for 'Thumbs Up'
            return "thumbs_up"
        elif landmarks[4][1] > landmarks[3][1]:  # Thumb rule for 'Thumbs Down'
            return "thumbs_down"
        elif all(landmarks[i][1] > landmarks[i - 1][1] for i in range(5, 21)):  # Closed fist
            return "fist"
        else:
            return "unknown"

    def process_frame(self, frame):
        """Processes each frame for hand detection and gesture recognition"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Extracting landmark positions
                height, width, _ = frame.shape
                landmarks = [
                    (int(landmark.x * width), int(landmark.y * height))
                    for landmark in hand_landmarks.landmark
                ]
                gesture = self.detect_gesture(landmarks)
                if gesture in self.gesture_map:
                    cv2.putText(frame, self.gesture_map[gesture], (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def run(self):
        """Starts the real-time hand gesture control interface"""
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.process_frame(frame)
            cv2.imshow("Hand Gesture Control", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    HandGestureControl().run()
