import os
import base64
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend


def derive_key_from_password(password: str, salt_file="salt.bin") -> bytes:
    """
    Derive a Fernet-compatible key from a password using PBKDF2-HMAC-SHA256.
    Stores/loads salt in 'salt.bin' so key is consistent across runs.
    """
    # Load or create salt
    if os.path.exists(salt_file):
        with open(salt_file, "rb") as f:
            salt = f.read()
    else:
        salt = os.urandom(16)
        with open(salt_file, "wb") as f:
            f.write(salt)

    # Derive 32-byte key
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=390000,
        backend=default_backend()
    )
    key = kdf.derive(password.encode("utf-8"))

    # Fernet requires urlsafe base64 encoding
    return base64.urlsafe_b64encode(key)
