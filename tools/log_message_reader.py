# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number], 
# held by Cisco Systems.
# -----------------------------------------------------------------------------

import asyncio
import dspy
import logging
from pydantic import BaseModel, PrivateAttr
from config import Settings
from lib.datastore import ShortTermMem
from lib.syslog_interface.syslog import SyslogAPI
from lib.syslog_interface.syslog_processor import SyslogProcessor


logger = logging.getLogger(__name__)
netio_logger = logging.getLogger("netio")


class LogMessageFirstResponder(dspy.Signature):
    """
    Analyze the incoming log messages according to the users question and then suggest to
    run a operational impact evaluation on them to find correlations, patterns, and
    clusters. Do not consider logs with a signature of %SEC_LOGIN-5-LOGIN_SUCCESS or
    %SYS-6-LOGOUT.
    """

    context: str = dspy.InputField(prefix="Context:", desc="log messages")
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="which operational impact evaluation to run")


class LogMessageProvider(dspy.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.analyze = dspy.ChainOfThought(LogMessageFirstResponder, max_retries=5)

    def forward(self, context, question):
        with dspy.context(lm=self.lm['gpt4o_mini']):
            next_action = self.analyze(context=context, question=question).answer
        return next_action


class LogMessageReader(BaseModel):
    lm: dict = None
    _processor: SyslogProcessor = PrivateAttr(default_factory=SyslogProcessor)
    _syslog_api: SyslogAPI = PrivateAttr(
        default_factory=lambda: SyslogAPI(api_base=Settings().syslog_endpoint.api_base)
    )

    def run(self, query: str) -> str:
        logger.debug(f"Running {self.__class__.__name__}")
        syslog = Settings().syslog_endpoint
        log_messages = self._syslog_api.get_syslog(syslog.host, syslog.buffer_name)
        context = asyncio.run(self._processor.run_workflow(dataset=log_messages, stream=True))
        ShortTermMem.set("context", context)
        next_action = LogMessageProvider(lm=self.lm)(context=context, question=query)
        return next_action
