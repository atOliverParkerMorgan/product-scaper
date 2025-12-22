from product_scraper.utils.console import (
    log_debug,
    log_error,
    log_info,
    log_success,
    log_warning,
)


def test_log_info(capsys):
    log_info("info message")
    out, _ = capsys.readouterr()
    assert "INFO:" in out


def test_log_success(capsys):
    log_success("success message")
    out, _ = capsys.readouterr()
    assert "SUCCESS:" in out


def test_log_warning(capsys):
    log_warning("warning message")
    out, _ = capsys.readouterr()
    assert "WARNING:" in out


def test_log_error(capsys):
    log_error("error message")
    out, _ = capsys.readouterr()
    assert "ERROR:" in out


def test_log_debug(capsys):
    log_debug("debug message")
    out, _ = capsys.readouterr()
    assert "DEBUG:" in out
