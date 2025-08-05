# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number], 
# held by Cisco Systems.
# -----------------------------------------------------------------------------

from abc import ABC, abstractmethod


class DataStoreInterface(ABC):
    @abstractmethod
    def get(self, key):
        """Get the value for the given key (can return None)."""
        pass

    @abstractmethod
    def set(self, key, value):
        """Set the value for the given key."""
        pass

    @abstractmethod
    def clear(self, key):
        """Clear the given key."""
        pass
