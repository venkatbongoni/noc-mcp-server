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
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, PrivateAttr
from lib.datastore import ShortTermMem
from lib.data_loader import DataLoader


logger = logging.getLogger(__name__)
netio_logger = logging.getLogger("netio")


class HolisticReasoner(dspy.Signature):
    """
    To perform a holistic reasoning assessment by integrating multiple inputs and
    understanding their interrelationships. You will start by examining the context
    (can be empty!) and evaluate and correlate any issues with any anomalies or
    irregularities found in the acquired diagnostic information or configuration from
    the nodes in the network. Your objective is to find the root cause of the problems
    in the logs. If there are no logs then you focus and reason about the acquired
    information with the objective to answer the question. You will formulate a
    detailed answer that articulates the root cause, explicitly mentioning 'node IP
    address' and 'affected IP address' and relevant data from logs or acquired
    information. Your analysis should demonstrate how the context informs your
    reasoning steps, how the question shapes your investigation, and how your reasoning
    leads to a justified and comprehensive conclusion.
    """

    context: List[str] = dspy.InputField(
        desc="log messages to evaluate",
    )
    question: str = dspy.InputField(
        desc="question to reason and answer",
    )
    acquired_information: List[str] = dspy.InputField(
        prefix="Diagnostic data from the nodes in the network:",
        desc="the information gathered from network nodes and to be used for holistic reasoning",
    )
    answer: str = dspy.OutputField(
        desc="holistic reasoning answer",
    )


class SyslogReasonerProvider(dspy.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.root_cause_assessment = dspy.ChainOfThought(HolisticReasoner, max_retries=5)

    def forward(self, context, question, acquired_information):
        logger.debug(f"Running {self.__class__.__name__}")
        with dspy.context(lm=self.lm["gpt4o_mini"]):
            causal_assessment = self.root_cause_assessment(
                context=context,
                question=question,
                acquired_information=acquired_information,
            )
        return causal_assessment


class SyslogHolisticReasoner(BaseModel):
    lm: dict = None
    model_path: Path = None
    _context: Optional[str] = None
    _dataloader = DataLoader()

    def run(self, query: str) -> tuple:
        logger.debug(f"Running {self.__class__.__name__}")
        context = ShortTermMem.get("high_impact") or ShortTermMem.get("context") or self._context or []
        exploration_output = ShortTermMem.get("explored_reasoning_outputs") or []
        reasoner = SyslogReasonerProvider(lm=self.lm)
        model = self.model_path / f"{reasoner.__class__.__name__}.json"

        if model.exists():
            reasoner.load(model)
            netio_logger.info(f" 🌡️ Loaded optimized actor from {model}")
        else:
            netio_logger.info(f" ❄️ Running unoptimized actor {reasoner.__class__.__name__}")

        causal_assessment = reasoner(
            context=context,
            question=query,
            acquired_information=exploration_output,
        )

        enable_data_sets = ShortTermMem.get("synthesize_actor_optimization_sets", None)
        if enable_data_sets:
            self._dataloader.actor_dataset_sampling(
                "SyslogReasonerProvider",
                context,
                query,
                exploration_output,
                causal_assessment.answer,
                max_train_samples=4,
                max_dev_samples=12,
            )
        return causal_assessment.answer, causal_assessment.reasoning
