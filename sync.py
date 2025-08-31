class EventSynchronizer:
    """
    Sync events with external storage or monitoring system.
    """

    def sync(self, event: dict):
        print(f"[SYNC] Event sent: {event}")

