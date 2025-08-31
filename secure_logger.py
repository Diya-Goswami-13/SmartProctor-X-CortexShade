import json
import time
from schemas import EventSchema
from crypto import CryptoUtils

class SecureLogger:
    """
    Logs validated events securely using encryption.
    """

    def __init__(self, crypto: CryptoUtils, logfile="secure_logs.jsonl"):
        self.crypto = crypto
        self.logfile = logfile
        self.schema = EventSchema()

    def log_event(self, event: dict):
        if not self.schema.validate(event):
            print("[SecureLogger] Invalid event schema. Skipping log.")
            return

        event["logged_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        serialized = json.dumps(event)
        encrypted = self.crypto.encrypt(serialized)

        with open(self.logfile, "ab") as f:
            f.write(encrypted + b"\n")

        print(f"[SecureLogger] Event logged securely at {event['logged_at']}")