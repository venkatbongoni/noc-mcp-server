# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number], 
# held by Cisco Systems.
# -----------------------------------------------------------------------------

from enum import Enum
from typing import Literal


ModeType = Literal["inference", "optimization", "offline_optimization"]


class OptimizationState(Enum):
    IS_OPTIMIZED = 'dspy_program_is_optimized'
    IS_NOT_OPTIMIZED = 'dspy_program_is_not_optimized'


class OptimizationBase:
    """
    Base class that provides optimization state management for agents.
    """
    def __init__(self):
        self.optimization_state: OptimizationState = OptimizationState.IS_NOT_OPTIMIZED

    def set_optimized(self):
        """
        Set the agent's optimization state to optimized.
        """
        self.optimization_state = OptimizationState.IS_OPTIMIZED

    def reset_optimization_state(self):
        """
        Reset the agent's optimization state to not optimized.
        """
        self.optimization_state = OptimizationState.IS_NOT_OPTIMIZED

    def is_optimized(self) -> bool:
        """
        Check if the agent is optimized.

        Returns:
            bool: True if the agent is optimized, False otherwise.
        """
        return self.optimization_state == OptimizationState.IS_OPTIMIZED
