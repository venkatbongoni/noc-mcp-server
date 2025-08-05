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
from pydantic import BaseModel, ValidationError, field_validator


class PatternParseError(Exception):
    """Raised when pattern parsing fails."""
    pass


class BasePatternModel(BaseModel):
    """Reusable base class for pattern-based string models constructed from logging data."""

    pattern: ClassVar[re.Pattern] = re.compile("")
    value: str
    __match_cache__: Optional[re.Match] = None

    @classmethod
    def try_parse(cls, s: str) -> Optional["BasePatternModel"]:
        try:
            return cls(value=s)
        except ValidationError:
            return None

    @classmethod
    def parse(cls, s: str) -> "BasePatternModel":
        try:
            return cls(value=s)
        except ValidationError as e:
            raise PatternParseError(f"Invalid {cls.__name__}: {s}") from e

    @field_validator("value")
    @classmethod
    def validate_format(cls, v):
        if not cls.pattern.search(v):
            raise ValueError(f"Invalid format for {cls.__name__}: {v}")
        return v

    @property
    def _match(self) -> Optional[re.Match]:
        if self.__match_cache__ is None:
            self.__match_cache__ = self.pattern.search(self.value)
        return self.__match_cache__

    def __repr__(self):
        return f"<{self.__class__.__name__} value={self.value}>"

    def __str__(self):
        return self.value
