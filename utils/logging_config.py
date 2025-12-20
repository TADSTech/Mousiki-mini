"""
Logging Configuration

Configures logging for the application.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from utils.config import settings


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def setup_logging():
    """Setup logging configuration."""
    
    # Create logs directory if it doesn't exist
    log_file = Path(settings.LOG_FILE)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Root logger configuration 
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Reduce noise from libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def success(msg: str, module: str = "APP"):
    """
    Log a success message with green colored output.
    
    Args:
        msg: Message to log
        module: Module name for context (default: "APP")
    
    Example:
        >>> from utils.logging_config import success
        >>> success("Database connected successfully", "DB")
    """
    print(f"{Colors.GREEN}{Colors.BOLD}✓ [{module}]{Colors.RESET} {Colors.GREEN}{msg}{Colors.RESET}")
    logger = get_logger(module)
    logger.info(f"SUCCESS: {msg}")


def warning(msg: str, module: str = "APP"):
    """
    Log a warning message with yellow colored output.
    
    Args:
        msg: Message to log
        module: Module name for context (default: "APP")
    
    Example:
        >>> from utils.logging_config import warning
        >>> warning("Cache miss, fetching from database", "CACHE")
    """
    print(f"{Colors.YELLOW}{Colors.BOLD}⚠ [{module}]{Colors.RESET} {Colors.YELLOW}{msg}{Colors.RESET}")
    logger = get_logger(module)
    logger.warning(msg)


def neutral(msg: str, module: str = "APP"):
    """
    Log a neutral/info message with blue colored output.
    
    Args:
        msg: Message to log
        module: Module name for context (default: "APP")
    
    Example:
        >>> from utils.logging_config import neutral
        >>> neutral("Starting recommendation engine", "ENGINE")
    """
    print(f"{Colors.BLUE}{Colors.BOLD}ℹ [{module}]{Colors.RESET} {Colors.BLUE}{msg}{Colors.RESET}")
    logger = get_logger(module)
    logger.info(msg)


def error(msg: str, module: str = "APP"):
    """
    Log an error message with red colored output.
    
    Args:
        msg: Message to log
        module: Module name for context (default: "APP")
    
    Example:
        >>> from utils.logging_config import error
        >>> error("Failed to connect to database", "DB")
    """
    print(f"{Colors.RED}{Colors.BOLD}✗ [{module}]{Colors.RESET} {Colors.RED}{msg}{Colors.RESET}")
    logger = get_logger(module)
    logger.error(msg)


def warninglog(msg: str):
    """Log a warning message."""
    logger = get_logger(__name__)
    logger.warning(msg)