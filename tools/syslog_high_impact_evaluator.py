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
from pydantic import BaseModel, PrivateAttr
from lib.datastore import ShortTermMem

logger = logging.getLogger(__name__)
netio_logger = logging.getLogger("netio")


class HighestImpactLogEvaluation(dspy.Signature):
    """
    Analyze all incoming log messages using triangulation techniques to identify patterns,
    correlations, and clusters. Select a subset of logs that most likely represent the root
    cause and/or key symptoms of a failure. Determine which of these issues has the highest
    operational impact based on severity, scope, and recurrence.
    """

    classified_logs: str = dspy.InputField(
        desc="classified network wide log messages",
    )
    question: str = dspy.InputField(
        desc="about classified logs",
    )
    high_impact_logs: str = dspy.OutputField(
        prefix="Exclusively Analyzing High Operational Impact logs:",
        desc="subset of the classified logs with the same JSON format",
    )


class HighestImpactProvider(dspy.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.log_evaluation = dspy.ChainOfThought(HighestImpactLogEvaluation, max_retries=5)

    def forward(self, logs, query):
        logger.debug(f"Running {self.__class__.__name__}")
        with dspy.context(lm=self.lm['gpt4o_mini']):
            high_impact = self.log_evaluation(classified_logs=logs, question=query).high_impact_logs
        return high_impact


class TriageSignature(dspy.Signature):
    """
    Analyze and triage the logs and determine the next action to take. The next action
    could be to run some commands on the remote system_ip address to gather more diagnostic
    data and perform further analysis.
    """

    logs: str = dspy.InputField(
        desc="high operational impact logs",
    )
    hint: str = dspy.InputField(
        desc="user instruction about what to do with the logs",
    )
    next_action: str = dspy.OutputField(
        prefix="Next Action:",
        desc="next action to take based on your triage results",
    )


class TriageActor(dspy.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.triage = dspy.ChainOfThought(TriageSignature, max_retries=5)

    def forward(self, logs, hint):
        logger.debug(f"Running {self.__class__.__name__}")
        with dspy.context(lm=self.lm['gpt4o_mini']):
            next_action = self.triage(logs=logs, hint=hint).next_action
        return next_action


class DecideHighestImpactIssue(BaseModel):
    lm: dict = None

    def run(self, query: str) -> str:
        logger.debug(f"Running {self.__class__.__name__}")
        classified = ShortTermMem.get("context", [])
        high_impact_logs = (
            None if not classified else HighestImpactProvider(lm=self.lm)(logs=classified, query=query)
        )
        ShortTermMem.set("high_impact", high_impact_logs)
        action = TriageActor(lm=self.lm)(logs=high_impact_logs, hint=query)
        return action
