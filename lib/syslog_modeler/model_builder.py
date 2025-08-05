# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number],
# held by Cisco Systems.
# -----------------------------------------------------------------------------

from collections import defaultdict
from datetime import datetime, timezone
from typing import List, Dict
from . import (
    CiscoSyslogModel,
    CiscoSyslogMnemonicModel,
    IOSXRDevicePathModel,
    InputMessageHeaderModel,
)
from .models import LogEntry, HardwarePath, Device, NetworkLogs, SyslogJsonOutput


class SyslogJsonOutputBuilder:
    """
    Builder for Syslog JSON output. Supports incremental/streaming ingestion
    (add_syslog_line), batch ingestion (add_syslog_lines) and final build().
    """

    def __init__(self):
        self.devices_data: Dict[str, Dict[str, List[LogEntry]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def add_syslog_line(self, line: str) -> "SyslogJsonOutputBuilder":
        """
        Ingest a single syslog line into the builder.
        """
        # Initialize all variables with default values
        message = "UNKNOWN"
        system_ip = "UNKNOWN"
        timestamp = datetime.now(timezone.utc).isoformat() + "Z"
        path = "DEFAULT"
        signature = "UNKNOWN"
        component = "UNKNOWN"
        severity = "UNKNOWN"

        # Parse the syslog line with different patterns
        log = CiscoSyslogModel.try_parse(line)
        header = InputMessageHeaderModel.try_parse(line)
        mnemonic = CiscoSyslogMnemonicModel.try_parse(line)
        hw_path = IOSXRDevicePathModel.try_parse(line)

        # Override defaults with parsed values if available
        if log:
            message = getattr(log, 'system_message', message)
        if header:
            system_ip = getattr(header, 'system_ip', system_ip)
            timestamp = getattr(header, 'timestamp', timestamp)
        if hw_path:
            path = getattr(hw_path, 'path', path)
        if mnemonic:
            signature = getattr(mnemonic, 'signature', signature)
            component = getattr(mnemonic, 'facility', component)
            severity_val = getattr(mnemonic, 'severity', None)
            if severity_val is not None:
                severity = str(severity_val)

        # Create a log entry and data ingest into the context model.
        self.devices_data[system_ip][path].append(
            LogEntry(
                timestamp=timestamp,
                signature=signature,
                component=component,
                severity=severity,
                message=message,
            )
        )

        return self

    def add_syslog_lines(self, lines: List[str]) -> "SyslogJsonOutputBuilder":
        """
        Ingest multiple syslog lines.
        """
        for line in lines:
            self.add_syslog_line(line)
        return self

    def build(self) -> SyslogJsonOutput:
        """
        Build the final SyslogJsonOutput model.
        """
        devices = []

        for system_ip, paths in self.devices_data.items():
            hardware_paths = {}

            for path, logs in paths.items():
                hardware_paths[path] = HardwarePath(logs=logs)

            devices.append(Device(
                system_ip=system_ip,
                hardware_path=hardware_paths
            ))

        return SyslogJsonOutput(network_logs=NetworkLogs(devices=devices))
