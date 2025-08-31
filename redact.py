class Redactor:
    """
    Removes sensitive info from text (simple demo).
    """

    def redact(self, text: str) -> str:
        return text.replace("secret", "[REDACTED]")

