import hashlib

class SnapshotHasher:
    """
    Creates secure hashes for snapshots or sensitive data.
    """

    def hash_snapshot(self, data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()

