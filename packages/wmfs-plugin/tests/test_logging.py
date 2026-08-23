from wmfs_plugin import LogLevel, NullLogger


def test_null_logger_is_nonthrowing_disabled_singleton() -> None:
    assert not NullLogger.enabled(LogLevel.DEBUG)
    assert NullLogger.bind(plugin="test") is NullLogger

    NullLogger.log(LogLevel.INFO, "message", "category", {"value": 1})
    NullLogger.debug("debug")
    NullLogger.info("info")
    NullLogger.warning("warning")
    NullLogger.error("error")
    NullLogger.critical("critical")
