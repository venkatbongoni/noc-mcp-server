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


class IOSXRDevicePathModel(BasePatternModel):
    """IOS-XR device path identifier, e.g. 'RP/0/RP0/CPU0', 'LC/0/0/CPU0', '0/0/0/CPU0'"""

    pattern: ClassVar[re.Pattern] = re.compile(r'\s([A-Z0-9]+/\d+/[A-Z0-9]+/[A-Z0-9]+)\b')

    @computed_field
    @property
    def path(self) -> Optional[str]:
        return self._match.group(1) if self._match else None
