# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number], 
# held by Cisco Systems.
# -----------------------------------------------------------------------------

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def file_logger_config(log_file_path: Path, name: str):
    """Function to configure logger for file logging."""
    file_logger = logging.getLogger(name)
    file_logger.setLevel(logging.INFO)

    if not file_logger.hasHandlers():
        handler = RotatingFileHandler(
            log_file_path, maxBytes=50 * 1024 * 1024, backupCount=5
        )  # Log file up to 50MB, keep 5 backups

        log_format = "%(asctime)s.%(msecs)03d - %(message)s"
        formatter = logging.Formatter(log_format, datefmt="%b %d %H:%M:%S")
        handler.setFormatter(formatter)
        file_logger.addHandler(handler)
    return file_logger
