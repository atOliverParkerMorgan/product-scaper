
"""
Console logging utilities for styled output using Rich.
"""

from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "bold red",
    "debug": "dim",
})

CONSOLE = Console(theme=custom_theme)

def log_info(message: str):
    """Log informational message in cyan"""
    CONSOLE.print(f"[info]INFO:[/info] {message}")

def log_success(message: str):
    """Log success message in green"""
    CONSOLE.print(f"[success]SUCCESS:[/success] {message}")

def log_warning(message: str):
    """Log warning message in yellow"""
    CONSOLE.print(f"[warning]WARNING:[/warning] {message}")

def log_error(message: str):
    """Log error message in red"""
    CONSOLE.print(f"[error]ERROR:[/error] {message}")

def log_debug(message: str):
    """Log debug message in dim"""
    CONSOLE.print(f"[debug]DEBUG:[/debug] {message}")
