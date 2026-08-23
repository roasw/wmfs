class OperationError(RuntimeError):
    """An operation failed without corrupting its worker session."""

    def __init__(self, error_type: str, message: str) -> None:
        self.error_type = error_type
        self.message = message
        super().__init__(f"{error_type}: {message}" if error_type else message)


class WorkerTransportError(RuntimeError):
    """A worker session can no longer be used safely."""
