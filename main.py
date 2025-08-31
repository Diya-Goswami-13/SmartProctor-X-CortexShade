import time
import json
import threading
import numpy as np
import subprocess
import keyboard

from key_manager import derive_key_from_password
from neurosentinel_camera import CameraDetector
import neurosentinel_audio as na

from schemas import EventSchema
from crypto import CryptoUtils
from secure_logger import SecureLogger
from snapshot_hasher import SnapshotHasher
from sync import EventSynchronizer
from retention import RetentionPolicy
from redact import Redactor


class CortexShadeSystem:
    def __init__(self, student_id="student1", password="default123"):
        self.student_id = student_id
        self.key = derive_key_from_password(password)
        self.crypto = CryptoUtils(key=self.key)
        self.logger = SecureLogger(crypto=self.crypto)
        self.schema = EventSchema()
        self.hasher = SnapshotHasher()
        self.syncer = EventSynchronizer()
        self.retention = RetentionPolicy()
        self.redactor = Redactor()
        self.camera = CameraDetector()
        self.camera.active = True
        self.suspicions = {"video": [], "audio": []}
        self.lock = threading.Lock()
        self.running = True

    def run_camera(self):
        while self.running:
            score = self.camera.run(student_id=self.student_id)
            if score > 0:
                event = {
                    "id": f"{self.student_id}_cam_{int(time.time())}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "data": f"Camera suspicion score: {score}"
                }
                event["data"] = self.redactor.redact(event["data"])
                self.logger.log_event(event)
                self.syncer.sync(event)
                with self.lock:
                    self.suspicions["video"].append(event)

    def run_audio(self):
        while self.running:
            try:
                audio, sr = na.record_audio(duration=5, fs=44100)
                features = na.extract_features(audio, sr)
                label, reasons = na.classify_audio(features)
                if label in ["whisper", "background_talk", "tapping", "normal_talk"]:
                    event = {
                        "id": f"{self.student_id}_aud_{int(time.time())}",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "data": f"Audio event: {label} | Reasons: {reasons}"
                    }
                    event["data"] = self.redactor.redact(event["data"])
                    self.logger.log_event(event)
                    self.syncer.sync(event)
                    with self.lock:
                        self.suspicions["audio"].append(event)
            except Exception as e:
                print(f"[Audio] Error: {e}")

    def encrypt_and_save(self):
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return str(obj)

        while self.running:
            time.sleep(10)
            with self.lock:
                with open("suspicions.json", "w") as f:
                    json.dump(self.suspicions, f, indent=2, default=convert)

                with open("suspicions.json", "rb") as f:
                    data = f.read()
                encrypted = self.crypto.encrypt(data.decode("utf-8"))

                with open("suspicions.json.encrypted", "wb") as f:
                    f.write(encrypted)

                print("[CortexShade] Suspicion log encrypted and saved.")

    def monitor_exit_keys(self):
        print("[CortexShade] Press Ctrl + Shift + Q to exit and launch admin panel.")
        while self.running:
            if keyboard.is_pressed("ctrl") and keyboard.is_pressed("shift") and keyboard.is_pressed("q"):
                print("\n[CortexShade] Shutdown signal received via Ctrl + Shift + Q.")
                self.running = False
                break
            time.sleep(0.5)

    def start(self):
        print("[CortexShade] Secure proctoring system initiated.")

        threading.Thread(target=self.run_camera, daemon=True).start()
        threading.Thread(target=self.run_audio, daemon=True).start()
        threading.Thread(target=self.encrypt_and_save, daemon=True).start()
        threading.Thread(target=self.monitor_exit_keys, daemon=True).start()

        while self.running:
            time.sleep(1)

        self.retention.apply()
        time.sleep(2)
        print("[CortexShade] System terminated securely.")
        print("[CortexShade] Launching admin panel...")
        subprocess.run(["python", "admin.py"])


if __name__ == "__main__":
    print("=== CortexShade Secure Proctoring ===")
    password = input("Enter encryption password: ")
    system = CortexShadeSystem(student_id="student1", password=password)
    system.start()