import cv2
import mediapipe as mp
import time
import os
import platform
import numpy as np


class CameraDetector:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[ERROR] Could not open camera.")
            exit(1)
        self.cheating_score = 0
        self.away_start = None
        self.active = False
        self.finger_alert_active = False

        if not os.path.exists("screenshots"):
            os.makedirs("screenshots")

    def get_head_direction(self, landmarks, w, h):
        nose = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        mouth = landmarks[13]

        nose_x, nose_y = int(nose.x * w), int(nose.y * h)
        left_x, left_y = int(left_eye.x * w), int(left_eye.y * h)
        right_x, right_y = int(right_eye.x * w), int(right_eye.y * h)
        mouth_y = int(mouth.y * h)

        face_center_x = (left_x + right_x) // 2
        eye_avg_y = (left_y + right_y) // 2
        offset_x = nose_x - face_center_x
        eye_to_nose = nose_y - eye_avg_y
        nose_to_mouth = mouth_y - nose_y

        if offset_x < -40:
            return "RIGHT"
        elif offset_x > 40:
            return "LEFT"

        if eye_to_nose < 15:
            return "UP"
        elif nose_to_mouth < 15:
            return "DOWN"

        return "CENTER"

    def count_fingers(self, hand_landmarks, handedness):
        tips_ids = [4, 8, 12, 16, 20]
        pip_ids = [2, 6, 10, 14, 18]
        fingers = []

        if handedness == "Right":
            if hand_landmarks.landmark[tips_ids[0]].x > hand_landmarks.landmark[pip_ids[0]].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[pip_ids[0]].x:
                fingers.append(1)
            else:
                fingers.append(0)

        for tip, pip in zip(tips_ids[1:], pip_ids[1:]):
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                fingers.append(1)
            else:
                fingers.append(0)
        return sum(fingers)

    def save_screenshot(self, frame, reason):
        filename = f"screenshots/{reason}_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"[ALERT] Screenshot saved: {filename}")

    def update_database(self, student_id, cheating_score):
        import json
        db_file = "cheating_scores.json"
        if os.path.exists(db_file):
            with open(db_file, "r") as f:
                data = json.load(f)
        else:
            data = {}
        data[student_id] = cheating_score
        with open(db_file, "w") as f:
            json.dump(data, f, indent=2)

    def run(self, student_id="student1"):
        try:
            winname = "Camera Detector"
            cv2.namedWindow(winname)
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                direction = "NO FACE"
                suspicious = False

                try:
                    _, _, win_w, win_h = cv2.getWindowImageRect(winname)
                except:
                    win_w, win_h = w, h

                if self.active and (win_w == 0 or win_h == 0):
                    warning_frame = np.zeros((200, 600, 3), dtype=np.uint8)
                    cv2.line(warning_frame, (0, 100), (600, 100), (0, 0, 255), 5)
                    cv2.putText(warning_frame, "WARNING: Camera Minimized!",
                                (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                    cv2.putText(warning_frame, f"Cheating Score: {self.cheating_score}",
                                (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(winname, warning_frame)
                    key = cv2.waitKey(100) & 0xFF
                    if key == 27:
                        print("[INFO] Exiting system...")
                        break
                    elif key == ord('p'):
                        print("[ADMIN] Camera paused and exiting.")
                        break
                    continue

                if self.active:

                    results = self.face_mesh.process(rgb)

                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        direction = self.get_head_direction(face_landmarks.landmark, w, h)

                        if direction in ["LEFT", "RIGHT", "UP", "DOWN"]:
                            if self.away_start is None:
                                self.away_start = time.time()
                            elif time.time() - self.away_start > 2:
                                self.cheating_score += 1
                                self.save_screenshot(frame, direction)
                                suspicious = True
                                self.away_start = None
                        else:
                            self.away_start = None


                    else:

                        if self.away_start is None:
                            self.away_start = time.time()
                        elif time.time() - self.away_start > 5:
                            self.cheating_score += 1
                            self.save_screenshot(frame, "NO_FACE")
                            suspicious = True
                            self.away_start = None

                    hand_results = self.hands.process(rgb)
                    finger_detected = False
                    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                        for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks,
                                                              hand_results.multi_handedness):
                            label = handedness.classification[0].label
                            fingers_up = self.count_fingers(hand_landmarks, label)
                            if fingers_up in [1, 2, 3, 4, 5]:
                                finger_detected = True

                            mp.solutions.drawing_utils.draw_landmarks(
                                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                            )

                    if finger_detected and not self.finger_alert_active:
                        self.cheating_score += 1
                        self.save_screenshot(frame, "fingers")
                        suspicious = True
                        self.finger_alert_active = True
                    elif not finger_detected:
                        self.finger_alert_active = False

                cv2.putText(frame, f"Admin Active: {self.active}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Direction: {direction}", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, f"Cheating Score: {self.cheating_score}", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if suspicious:
                    cv2.rectangle(frame, (20, 160), (w - 20, 220), (0, 0, 255), -1)
                    cv2.putText(frame, "ALERT: Suspicious Behavior Detected!",
                                (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 3)

                cv2.imshow(winname, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("[INFO] Exiting system...")
                    break
                elif key == ord('s'):
                    self.active = True
                    print("[ADMIN] Camera started.")
                elif key == ord('p'):
                    print("[ADMIN] Camera paused and exiting.")
                    break

        finally:
            if self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Camera released & windows closed.")
            self.update_database(student_id, self.cheating_score)
            print(f"Final Cheating Score for {student_id}: {self.cheating_score}")

        return self.cheating_score


if __name__ == "__main__":
    detector = CameraDetector()
    detector.run()