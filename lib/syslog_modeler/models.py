# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number], 
# held by Cisco Systems.
# -----------------------------------------------------------------------------

from pydantic import BaseModel, Field
from typing import List, Dict


class LogEntry(BaseModel):
    """Individual log entry with timestamp, severity, facility, and message."""
    timestamp: str
    signature: str
    component: str
    severity: str
    message: str


class HardwarePath(BaseModel):
    """Contains logs for a specific hardware path."""
    logs: List[LogEntry] = Field(default_factory=list)


class Device(BaseModel):
    """Represents a network device with its properties and hardware paths."""
    system_ip: str
    hardware_path: Dict[str, HardwarePath]


class NetworkLogs(BaseModel):
    """Top-level structure containing the list of devices."""
    devices: List[Device] = Field(default_factory=list)


class SyslogJsonOutput(BaseModel):
    """Root structure for the JSON output."""
    network_logs: NetworkLogs = Field(default_factory=NetworkLogs)
