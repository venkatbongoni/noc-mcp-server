# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number], 
# held by Cisco Systems.
# -----------------------------------------------------------------------------
import re
from typing import Optional, ClassVar
from pydantic import computed_field
from .base import BasePatternModel


class CiscoSyslogModel(BasePatternModel):
    """Cisco syslog message defined by '%'."""

    pattern: ClassVar[re.Pattern] = re.compile(r'%[A-Z][A-Z0-9_-]*-\d+-[A-Z0-9_-]+\s*:\s+.+')

    @computed_field
    @property
    def system_message(self) -> Optional[str]:
        return self._match.group(0) if self._match else None


class CiscoSyslogMnemonicModel(BasePatternModel):
    """Cisco syslog mnemonic model, e.g. '%OSPF-4-FLOOD_WAR', '%OS-XTC-5-SR_POLICY_UPDOWN'"""

    pattern: ClassVar[re.Pattern] = re.compile(r'%([A-Z][A-Z0-9_-]*)-(\d+)-([A-Z0-9_-]+)\b')

    @computed_field
    @property
    def signature(self) -> Optional[str]:
        return self._match.group(0) if self._match else None

    @computed_field
    @property
    def facility(self) -> Optional[str]:
        return self._match.group(1) if self._match else None

    @computed_field
    @property
    def severity(self) -> Optional[int]:
        return int(self._match.group(2)) if self._match else None

    @computed_field
    @property
    def mnemonic(self) -> Optional[str]:
        return self._match.group(3) if self._match else None


class InputMessageHeaderModel(BasePatternModel):
    """Syslog added message header, e.g. 'Aug 29 13:46:01 10.90.91.41 102', where the IP
    address is the system IP originating the message and the number is the message-ID."""

    pattern: ClassVar[re.Pattern] = re.compile(r'^([A-Za-z]{3}\s+\d{1,2}\s\d{2}:\d{2}:\d{2})\s(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\s(\d+)\b')

    @computed_field
    @property
    def timestamp(self) -> Optional[str]:
        return self._match.group(1) if self._match else None

    @computed_field
    @property
    def system_ip(self) -> Optional[str]:
        return self._match.group(2) if self._match else None

    @computed_field
    @property
    def message_id(self) -> Optional[str]:
        return self._match.group(3) if self._match else None
