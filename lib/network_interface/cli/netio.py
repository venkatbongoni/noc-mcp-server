# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number],
# held by Cisco Systems.
# -----------------------------------------------------------------------------

import json
import logging
import requests
import ipaddress
from datetime import datetime, timezone
from typing import List
from config import Settings
from tools.lib.base_models import CommandOutputModel, DeviceCommand, ReasoningStepEntry
from lib.datastore import ShortTermMem
from .modules import (
    CommandVerificationProvider,
    ShowCommandProviderOpenAI,
    # ShowCommandProviderLlama3,
    # ShowCommandProviderRAG,
)


logger = logging.getLogger(__name__)
netio_logger = logging.getLogger("netio")


class DeviceCommandLineAPI:
    """
    Network device command line API for use with Agent Tools.
    """

    def __init__(self, lm: dict):
        self.lm = lm

    def _get_json_response(self, data: dict) -> CommandOutputModel:
        return CommandOutputModel(**data)

    def _command_execution(self, command: DeviceCommand) -> str:
        command_output_lst = []
        data = None
        command_api_base = Settings().command_endpoint.api_base
        logger.debug(f"Running {self.__class__.__name__}._command_execution()")

        command_dict = command.model_dump()
        if isinstance(
            command_dict.get("ip_address"),
            (ipaddress.IPv4Address, ipaddress.IPv6Address),
        ):
            command_dict["ip_address"] = str(command_dict["ip_address"])

        try:
            response = requests.post(
                command_api_base,
                json=command_dict,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()

        # Handle HTTP errors as reward signals, as implemented by the Network API.
        except requests.HTTPError as e:
            # 504 indicates that we have sent a syntactically incorrect command.
            if e.response is not None and e.response.status_code == 504:
                return ""
            # 413 indicates that the response was larger than limits defined by Network API.
            elif e.response is not None and e.response.status_code == 413:
                f" ❌ Network API returns content too large error for '{command.command}' on '{command.ip_address}'. "
                return ""
            else:
                netio_logger.warning(
                    f" ❌ Failed to execute command '{command.command}' on '{command.ip_address}'. "
                    f"HTTPError: {e}",
                )
                return ""
        except requests.RequestException as e:
            netio_logger.warning(
                f" ❌ Failed to execute command '{command.command}' on '{command.ip_address}'. "
                f"Exception: {e}",
            )
            return ""

        try:
            command_output = response.json()
        except json.JSONDecodeError as json_err:
            netio_logger.warning(
                f" ❌ Invalid JSON response for command '{command.command}' on '{command.ip_address}'. "
                f"Exception: {json_err}\nResponse content: {response.text}",
            )
            return ""

        output_text = command_output.get("output", "")
        if "Invalid input detected" in output_text:
            netio_logger.warning(
                f" ❌ Invalid command detected for '{command.ip_address}': '{command.command}'."
            )
            return ""
        else:
            data = {
                "node_ip_address": command_dict["ip_address"],
                "command_output": output_text.splitlines(),
            }

        if not data:
            data = {"node_ip_address": "", "command_output": []}
        command_output_lst.append(data)
        node_command_model = [self._get_json_response(node_data) for node_data in command_output_lst]
        json_output = json.dumps([model.model_dump() for model in node_command_model], indent=2)
        return json_output

    def _detect_cisco_software(self, show_version_output: str) -> str:
        text = show_version_output.lower()
        if "ios xr software" in text:
            return "IOS XR"
        if "ios xe software" in text:
            return "IOS XE"
        if "nexus operating system (nx-os)" in text or "nx-os software" in text:
            return "NX-OS"
        # Classic IOS (must check after XE/XR)
        if "ios software" in text:
            return "IOS"
        return ""

    def _detect_platform_software_type(self, device: DeviceCommand) -> str:
        logger.debug(f"Running {self.__class__.__name__}._detect_platform_software_type()")
        json_output = self._command_execution(device)
        try:
            show_version_data = json.loads(json_output)[0]
            show_version_output = show_version_data.get("command_output", [])
            if not show_version_output:
                netio_logger.warning(f" ❌ {device.ip_address} no output from {device.command} command.")
                return ""
            software_type = self._detect_cisco_software("\n".join(show_version_output))
            if not software_type:
                netio_logger.warning(f" ❌ {device.ip_address} could not detect platform software type.")
                return ""
            netio_logger.info(f" ✔️ {device.ip_address} detected platform software type: {software_type}")
        except (json.JSONDecodeError, ValueError):
            netio_logger.warning(f" ❌ {device.ip_address} failed to detect device platform, software, type.")
            return ""
        return software_type

    def _show_command_pipeline(self, action: str) -> List[DeviceCommand]:
        logger.debug(f"Running {self.__class__.__name__}._show_command_pipeline()")
        exploration_list = []

        # proposed_commands = ShowCommandProviderRAG()(action=action)
        # proposed_commands = ShowCommandProviderLlama3(lm=self.lm)(action=action)
        proposed_commands = ShowCommandProviderOpenAI(lm=self.lm)(action=action)
        if not proposed_commands:
            netio_logger.warning(" ❌ No commands extracted from actions description.")
            return exploration_list

        netio_logger.info(f" 💭 Proposed command: '{proposed_commands}'")
        for command in proposed_commands:
            if not isinstance(command, DeviceCommand):
                netio_logger.warning(f" ❌ Invalid command type: {type(command)}. Expected DeviceCommand.")
                continue

            # Identify the network device platform type
            cisco_command = DeviceCommand(ip_address=command.ip_address, command="show version | include Software")
            platform_type = self._detect_platform_software_type(cisco_command)
            if not platform_type:
                device_key = f"{str(cisco_command.ip_address)}:{cisco_command.command}"
                feedback_entry = ReasoningStepEntry(
                    command=cisco_command.command,
                    device_key=device_key,
                    status="failed",
                    info="No data received from device. Likely inaccessible or no Cisco platform. Do not retry.",
                    timestamp=datetime.now(timezone.utc).replace(microsecond=0)
                )

                # Update simple policy model
                ip_key = str(cisco_command.ip_address)

                def add_negative_feedback(neg_fb):
                    d = neg_fb or {}
                    d.setdefault(ip_key, []).append(feedback_entry)
                    return d

                ShortTermMem.update("negative_feedback", add_negative_feedback, default={})
                continue

            verified_command = CommandVerificationProvider(lm=self.lm)(
                proposed_command=command, cisco_os_software_type=platform_type
            )
            if not verified_command:
                netio_logger.warning(f" ❌ Command verification failed for: {command}")
                continue

            netio_logger.info(f" ✔️ Verified command: {verified_command}")
            exploration_list.append(verified_command)
        return exploration_list

    def _command_execution_pipeline(self, command: DeviceCommand) -> str:
        logger.debug(f"Running {self.__class__.__name__}._command_execution_pipeline()")
        netio_logger.info(f" ⚡ Command request (to network API): '{command}'")
        json_output = self._command_execution(command)
        netio_logger.info(f" ⚡ Command response (from network API): '{json_output}'")
        return json_output
