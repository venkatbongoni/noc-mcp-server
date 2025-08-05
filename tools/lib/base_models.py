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
from datetime import datetime
from typing import List, Literal, Any, Annotated, Optional
from pydantic import BaseModel, Field, IPvAnyAddress


logger = logging.getLogger(__name__)


class DeviceCommand(BaseModel):
    ip_address: IPvAnyAddress = Field(..., description="a valid system IP address")
    command: str = Field(..., description="one single command")


class CommandOutputModel(BaseModel):
    node_ip_address: str = Field(..., description="a valid system IP address")
    command_output: List[str] = Field(..., description="one single command output")


class ExplorationOutputEntry(BaseModel):
    node_ip_address: str = Field(..., description="IP address of the network node")
    command_output: List[str] = Field(..., description="List of lines from command output or feedback")


class ReasoningStepEntry(BaseModel):
    command: str = Field(..., description="Executed device command")
    device_key: str = Field(..., description="Composite key of device IP and command")
    status: str = Field(..., description="Execution status ('success' or 'failed')")
    info: str = Field(..., description="Additional information about execution result")
    timestamp: datetime = Field(..., description="ISO-formatted UTC timestamp of execution")


class SeverityLevelMessages(BaseModel):
    system_ip: IPvAnyAddress = Field(..., description="The IP address where the log originated from")
    device_hardware_path: Optional[Annotated[str, Field(pattern=r"\s[A-Z0-9]+/\d+/[A-Z0-9]+/[A-Z0-9]+\b")]] = Field(
        None,
        description="IOS-XR device path identifier, e.g. 'RP/0/RP0/CPU0', 'LC/0/0/CPU0', '0/0/0/CPU0'"
    )
    process_identifier: Optional[Annotated[str, Field(pattern=r"[A-Za-z0-9_]+\[\d+\]")]] = Field(
        None,
        description="Process identifier like 'xtc_agent[1304]' or 'SSHD_[67384]'"
    )
    syslog_component: str = Field(
        ...,
        pattern=r"%[A-Z][A-Z0-9_-]*-\d+-[A-Z0-9_-]+\b",
        description="Cisco syslog mnemonic model '%COMPONENT-SEVERITY-MNEMONIC', e.g. '%OSPF-4-FLOOD_WAR', '%OS-XTC-5-SR_POLICY_UPDOWN'"
    )
    severity_level: int = Field(ge=0, le=5)
    syslog_definition: Literal[
        "LOG_EMERG",
        "LOG_ALERT",
        "LOG_CRIT",
        "LOG_ERR",
        "LOG_WARNING",
        "LOG_NOTICE",
        "LOG_INFO",
        "LOG_DEBUG",
    ]
    log_messages: List[str]


class LogDataInterpretation(dspy.Signature):
    """
    As a Data Interpretation Actor, your task is to read and interpret structured
    logging data from devices, including tables and key-value data where keys are
    strings (sometimes containing colons) and values are numbers (sometimes a word in
    the context of the key). You are to analyze this data to extract meaningful
    insights for each node IP address, identify any anomalies or irregularities, and
    provide a clear summary of your findings relevant to the given hinted objective and
    advise a next step towards resolution. If there is no log data then respond
    accordingly.
    """

    structured_log_data: List[Any] = dspy.InputField(
        prefix="Structured log data:",
        desc="focussed log data from node IP address",
    )
    hint: str = dspy.InputField(
        desc="hinting a command when there is no diagnostic data provided",
    )
    summary: str = dspy.OutputField(
        prefix="The next exploration action:",
        desc="factoid summary including the 'On node ..' IP address OR Finish if no log data",
    )


class LogDataInterpretationActor(dspy.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.data_interpretation = dspy.ChainOfThought(LogDataInterpretation, max_retries=5)

    def forward(self, structured_outputs, hint):
        logger.debug(f"Running {self.__class__.__name__}")
        with dspy.context(lm=self.lm["gpt4o_mini"]):
            next_action = self.data_interpretation(structured_log_data=structured_outputs, hint=hint).summary
        return next_action
