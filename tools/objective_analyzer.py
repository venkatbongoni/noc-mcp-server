# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number], 
# held by Cisco Systems.
# -----------------------------------------------------------------------------

import dspy
import logging
from typing import Optional, List, Dict
from pydantic import BaseModel
from lib.datastore import ShortTermMem
from .lib.base_models import ReasoningStepEntry


logger = logging.getLogger(__name__)
netio_logger = logging.getLogger("netio")


class NewTrajectorySignature(dspy.Signature):
    """
    Check the current progress and determine whether the user's query is a question or a
    direct command. If it is a command or request for an action (e.g., "pull the logs",
    "show interfaces"), execute the relevant tool once, provide the output to the user, and
    finish. If it is a question that requires reasoning or explanation, check if you have
    enough data to fully answer. If yes, provide the answer. If not, plan additional
    exploration to gather the required data.
    """

    context: List[str] = dspy.InputField(
        desc="these log messages are the reason or evidence for why this objective was chosen",
    )
    current_progress: str = dspy.InputField(
        desc="given the current progress and comparing to the global objective",
    )
    user_query: str = dspy.InputField(
        desc="the last question from the user in the current conversation",
    )
    explored_reasoning_steps: Dict[str, List[ReasoningStepEntry]] = dspy.InputField(
        desc="the exploration steps successfully executed so far",
    )
    negative_feedback: Dict[str, List[ReasoningStepEntry]] = dspy.InputField(
        desc="prohibited next steps",
    )
    action: str = dspy.OutputField(
        prefix="Next network exploration:",
        desc=(
            "Finish, the analysis is complete, the result is ...   OR verbose describe "
            "the next action with IP address and exact command"
        )
    )


class NewTrajectory(dspy.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.explore_next_step = dspy.ChainOfThought(NewTrajectorySignature, max_retries=5)

    def forward(self, context, current_progress, user_query, explored_paths, negative_feedback):
        logger.debug(f"Running {self.__class__.__name__}")
        with dspy.context(lm=self.lm["gpt4o_mini"]):
            explore_next_step = self.explore_next_step(
                context=context,
                current_progress=current_progress,
                user_query=user_query,
                explored_reasoning_steps=explored_paths,
                negative_feedback=negative_feedback,
            )
        return explore_next_step.action, explore_next_step.reasoning


class AnalyzeTheObjectiveFulfillment(BaseModel):
    lm: dict = None
    _context: Optional[List] = None

    def __init__(self, **data):
        super().__init__(**data)

    def run(self, query: str) -> tuple[str, str]:
        logger.debug(f"Running {self.__class__.__name__}")
        context = ShortTermMem.get("high_impact") or ShortTermMem.get("context") or self._context or []
        user_query = ShortTermMem.get("user_query")

        action, reasoning = NewTrajectory(lm=self.lm)(
            context=context,
            current_progress=query,
            user_query=user_query,
            explored_paths=ShortTermMem.get("explored_reasoning_steps"),
            negative_feedback=ShortTermMem.get("negative_feedback"),
        )
        netio_logger.info(f" 👣 Next step: '{action}'")
        return action, reasoning
