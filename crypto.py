from cryptography.fernet import Fernet

class CryptoUtils:
    """
    Handles encryption and decryption using Fernet symmetric encryption.
    Accepts an external key for integration with password-derived keys.
    """

    def __init__(self, key=None):
        # Use provided key or generate a new one
        self.key = key or Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt(self, text: str) -> bytes:
        """
        Encrypts a string and returns the encrypted bytes.
        """
        return self.cipher.encrypt(text.encode())

    def decrypt(self, token: bytes) -> str:
        """
        Decrypts encrypted bytes and returns the original string.
        """
        return self.cipher.decrypt(token).decode()