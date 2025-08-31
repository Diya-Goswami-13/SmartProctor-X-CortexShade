import json
import os
import cv2
from key_manager import derive_key_from_password
from crypto import CryptoUtils


def decrypt_suspicions(password: str, encrypted_file="suspicions.json.encrypted"):
    """
    Decrypts the suspicion log, displays screenshots, and calculates the net cheating score.
    """
    try:
        # Derive key and decrypt
        key = derive_key_from_password(password)
        crypto = CryptoUtils(key=key)

        with open(encrypted_file, "rb") as f:
            encrypted_data = f.read()

        decrypted_text = crypto.decrypt(encrypted_data)
        data = json.loads(decrypted_text)

        print("\n=== Decrypted Suspicion Log ===")
        print(json.dumps(data, indent=2))

        # Calculate scores
        # Count video score based on number of screenshots
        video_score = 0
        if os.path.exists("screenshots"):
            video_score = len([f for f in os.listdir("screenshots") if f.endswith(".jpg")])

        # Count audio score based on number of entries in the 'audio' list
        audio_score = len(data.get("audio", []))

        net_cheating_score = video_score + audio_score

        print("\n=== Cheating Scores ===")
        print(f"Video Score (based on screenshots): {video_score}")
        print(f"Audio Score (based on log entries): {audio_score}")
        print(f"Net Cheating Score: {net_cheating_score}")

        # Display screenshots from video events
        print("\n=== Displaying Screenshots ===")
        if video_score > 0:
            for event in data.get("video", []):
                reason = event.get("event", "unknown")
                timestamp = event.get("timestamp", "unknown")

                # Match screenshot filename pattern
                screenshots = [
                    f for f in os.listdir("screenshots")
                    if reason in f and f.endswith(".jpg")
                ]
                for file in screenshots:
                    img = cv2.imread(os.path.join("screenshots", file))
                    if img is not None:
                        cv2.imshow(f"{reason} @ {timestamp}", img)
                        cv2.waitKey(1500)  # Show for 1.5 seconds
                        cv2.destroyAllWindows()
        else:
            print("No screenshots found.")

    except FileNotFoundError:
        print(f"[ERROR] Encrypted file '{encrypted_file}' or 'screenshots' directory not found.")
    except Exception as e:
        print(f"[ERROR] Failed to decrypt or display screenshots: {e}")


def main():
    print("=== CortexShade Admin Panel ===")
    password = input("Enter password to decrypt suspicion log: ")
    decrypt_suspicions(password)


if __name__ == "__main__":
    main()