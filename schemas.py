class EventSchema:
    """
    Validates event data.
    Example: {"id": "123", "timestamp": "2025-08-29", "data": "secret"}
    """

    def validate(self, event: dict) -> bool:
        required_keys = ["id", "timestamp", "data"]
        return all(key in event for key in required_keys)
