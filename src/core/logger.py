import logging
import logging.handlers
import os
from pathlib import Path
from .config import get_config

def get_logger(name: str, security_log: bool = False):
    config = get_config()
    logger = logging.getLogger(name)
    
    # Set log level from config
    log_level = getattr(logging, config.logging.log_level.upper())
    logger.setLevel(log_level)

    # Create log directory if it doesn't exist
    log_dir = Path(config.logging.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Console handler setup (only for non-security logs)
    if config.logging.console_logging and not security_log:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(config.logging.log_format))
        logger.addHandler(console_handler)

    # File handler setup
    if security_log:
        log_file = log_dir / "security_results.log"
        formatter = logging.Formatter('%(asctime)s - %(message)s')
    else:
        log_file = log_dir / "rocklm.log"
        formatter = logging.Formatter(config.logging.log_format)

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config.logging.max_log_size_mb * 1024 * 1024,
        backupCount=config.logging.backup_count
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
