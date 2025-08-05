# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number], 
# held by Cisco Systems.
# -----------------------------------------------------------------------------

from threading import Lock
from collections import defaultdict
from typing import Callable, Any
from .datastore_interface import DataStoreInterface


class LocalDataStore(DataStoreInterface):
    _instance = None
    _lock = Lock()
    _data: defaultdict = None

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._data = defaultdict(lambda: None)
            return cls._instance

    def get(self, key: str, default=None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def clear(self, key: str) -> None:
        with self._lock:
            if key in self._data:
                del self._data[key]

    def update(self, key: str, update_fn: Callable, default=None) -> Any:
        with self._lock:
            current = self._data.get(key, default)
            new_value = update_fn(current)
            self._data[key] = new_value
            return new_value

    def __getitem__(self, key):
        with self._lock:
            return self._data[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._data[key] = value

    def __delitem__(self, key):
        with self._lock:
            del self._data[key]

    def __iter__(self):
        with self._lock:
            return iter(self._data.copy())

    def __len__(self):
        with self._lock:
            return len(self._data)

    def keys(self):
        with self._lock:
            return list(self._data.keys())

    def values(self):
        with self._lock:
            return list(self._data.values())

    def items(self):
        with self._lock:
            return list(self._data.items())

    def get_all_data(self):
        with self._lock:
            return dict(self._data)

    def clear_data(self):
        with self._lock:
            self._data.clear()


ShortTermMem = LocalDataStore()
