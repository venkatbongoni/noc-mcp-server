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
from .signatures import (
    ActionToCommand,
    CommandAdapter,
    CommandRAG,
    DeviceCommandVerification,
)


logger = logging.getLogger(__name__)
netio_logger = logging.getLogger("netio")


class CommandAdapterProvider(dspy.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.command_adapter = dspy.ChainOfThought(CommandAdapter, max_retries=5)

    def forward(self, unstructured_commands):
        logger.debug(f"Running {self.__class__.__name__}")
        with dspy.context(lm=self.lm['gpt4o_mini']):
            commands = self.command_adapter(unstructured_commands=unstructured_commands).structured_commands
        return commands


class ShowCommandProviderOpenAI(dspy.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.action_to_command = dspy.ChainOfThought(ActionToCommand, max_retries=5)

    def forward(self, action):
        logger.debug(f"Running {self.__class__.__name__}")
        with dspy.context(lm=self.lm['gpt4o_mini']):
            commands = self.action_to_command(action=action).device_commands

        netio_logger.info(f"Synthesized commands: '{commands}'")
        # device_commands = CommandAdapterProvider(lm=self.lm)(unstructured_commands=commands)
        return commands


class ShowCommandProviderLlama3(dspy.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.action_to_command = dspy.ChainOfThought(ActionToCommand, max_retries=5)

    def forward(self, action):
        logger.debug(f"Running {self.__class__.__name__}")
        stop_token = '<|eot_id|>'  # Llama3 stop token with vLLM https://github.com/vllm-project/vllm/issues/4180
        action += stop_token
        with dspy.context(lm=self.lm['cisco_show_commands']):
            device_commands = self.action_to_command(command=action).device_commands
        netio_logger.info(f"Synthesized commands: '{device_commands}'")
        device_commands = CommandAdapterProvider(lm=self.lm)(unstructured_commands=device_commands)
        return device_commands


class ShowCommandProviderRAG(dspy.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.retrieve = dspy.Retrieve(k=3)
        self.generate_answer = dspy.ChainOfThought(CommandRAG)

    def forward(self, action):
        logger.debug(f"Running {self.__class__.__name__}")
        context = self.retrieve(action).passages
        device_commands = self.generate_answer(context=context, question=action).answer
        netio_logger.info(f"Synthesized commands: '{device_commands}'")
        device_commands = CommandAdapterProvider(lm=self.lm)(unstructured_commands=device_commands)
        return device_commands


class CommandVerificationProvider(dspy.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.command_verification = dspy.ChainOfThought(DeviceCommandVerification, max_retries=5)

    def forward(self, proposed_command, cisco_os_software_type="IOS"):
        logger.debug(f"Running {self.__class__.__name__}")
        with dspy.context(lm=self.lm['gpt4o_mini']):
            command = self.command_verification(proposed_command=proposed_command,
                                                cisco_os_software_type=cisco_os_software_type).verified_command
        return command
