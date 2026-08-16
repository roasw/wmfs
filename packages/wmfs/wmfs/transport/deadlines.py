import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TransportDeadlines:
    startup: float = 30.0
    request: float = 30.0
    fd_transfer: float = 5.0
    shutdown: float = 30.0
    kill_grace: float = 30.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{name} deadline must be a finite positive number"
                ) from error
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} deadline must be a finite positive number")
            object.__setattr__(self, name, value)


DEFAULT_TRANSPORT_DEADLINES = TransportDeadlines()
