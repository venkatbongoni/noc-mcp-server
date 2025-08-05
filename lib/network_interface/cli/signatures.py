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
from typing import List
from tools.lib.base_models import DeviceCommand

netio_logger = logging.getLogger("netio")


class CommandAdapter(dspy.Signature):
    """
    To extract the IP address and corresponding command (always next to each other) from
    the unstructured input and transform it into a list of tuples of the IP address and
    corresponding command. You need to make sure that the following command restrictions
    apply. Do NOT transform any debug commands, for example: 'debug IP packets'.
    """
    unstructured_commands: str = dspy.InputField(
        desc="IP address and command, comma separated",
    )
    structured_commands: List[DeviceCommand] = dspy.OutputField(
        prefix="Commands:",
        desc="list of tuples IP address and command",
    )


class CommandRAG(dspy.Signature):
    """
    To transform the context description under the objective of the question (intention)
    into the Cisco show command with the correct syntax so that it can be executed on the
    remote device IP address without any error. Identify the IP address of the network node
    where this command should be run.
    """
    context: str = dspy.InputField(
        desc="text description about cisco commands to be used for this specific task"
    )
    question: str = dspy.InputField(
        desc="the intention to use this command but the syntax may be wrong"
    )
    answer: str = dspy.OutputField(
        desc="short factoid comma separated answer containing remote IP address and the show command"
    )


class ActionToCommand(dspy.Signature):
    """
    You are given `Action`, a description that reference network device IP addresses and
    specify what needs to be checked on each device. Your task is to: Extract every IP
    address mentioned in the `Action`, along with the specific verification or
    troubleshooting step described for each device. For each device, generate the exact
    Cisco IOS command(s) necessary to perform the described check or analysis. Return your
    results as a list of tuples, where each tuple contains: The network device IP address,
    The Cisco IOS command that should be executed on that device. Ensure that each command
    is mapped to the correct device IP, and that the command syntax follows Cisco IOS
    conventions. This mapping will be used to explore the network and perform automated
    network diagnostics and facilitate root cause analysis of potential software or
    operational failures. Command Restrictions: Do not use debug-level commands under any
    circumstances. You may, however, use filtering commands with a pipe, for example
    `| include`, but only when the expected output is very large.
    """
    action: str = dspy.InputField(
        prefix="Action:",
        desc="description of the actions to be performed on one or many network devices",
    )
    device_commands: List[DeviceCommand] = dspy.OutputField(
        prefix="Correct Syntax:",
        desc="list of tuples of 'remote IP address' and 'command'",
    )


class DeviceCommandVerification(dspy.Signature):
    """
    To ensure that the command syntax is compatible with the specified Software Platform
    Type. Adjust the command syntax as necessary so that it matches the requirements of
    the identified Software Platform Type.
    """
    proposed_command: DeviceCommand = dspy.InputField(
        desc="A tuple of IP address and command to be verified",
    )
    cisco_os_software_type: str = dspy.InputField(
        prefix="Software Platform Type:",
        desc="Cisco OS/software type, for example: 'IOS XE', 'IOS XR', 'NX-OS', 'IOS'",
    )
    verified_command: DeviceCommand = dspy.OutputField(
        desc="A tuple of 'remote IP address' and platform verified 'command'",
    )


class BestAnswer(dspy.Signature):
    """Select the best answer."""
    question: str = dspy.InputField()
    device_command: List[DeviceCommand] = dspy.OutputField(
        desc="list of tuples of 'remote IP address' and 'command'",
    )
