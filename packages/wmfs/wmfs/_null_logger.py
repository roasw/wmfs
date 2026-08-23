class _NullLogger:
    __slots__ = ()

    def enabled(self, level: object) -> bool:
        return False

    def log(self, *args: object, **kwargs: object) -> None:
        return None

    debug = log
    info = log
    warning = log
    error = log
    critical = log

    def bind(self, **fields: object) -> "_NullLogger":
        return self


NULL_LOGGER = _NullLogger()
