# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number], 
# held by Cisco Systems.
# -----------------------------------------------------------------------------

import re
import dspy
import logging
import json
import ipaddress
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Set, Tuple
from pydantic import BaseModel, PrivateAttr
from lib.datastore import ShortTermMem
from lib.network_interface.cli.netio import DeviceCommandLineAPI
from .lib.base_models import ReasoningStepEntry, ExplorationOutputEntry, DeviceCommand


logger = logging.getLogger(__name__)
netio_logger = logging.getLogger("netio")


class NextStepPlanning(dspy.Signature):
    """
    To triage the acquired diagnostic data under the given objective and correlate these
    with the anomalies in the log messages to find a root cause. If the acquired diagnostic
    data does not reveal an issue, consider it insufficient. Adjust your exploration plan
    by identifying the next feasible combination of node IP address and command to gather
    more data towards finding a root cause. If the objective has been fulfilled then stop
    your exploration.
    """

    context: List[str] = dspy.InputField(
        desc="logging messages",
    )
    objective: str = dspy.InputField(
        desc="sub-objective to be considered for planning the network exploration steps",
    )
    acquired_information: Dict[str, List[ExplorationOutputEntry]] = dspy.InputField(
        prefix="Gathered diagnostic data:",
        desc="information to be used for the next step analysis",
    )
    explored_reasoning_steps: Dict[str, List[ReasoningStepEntry]] = dspy.InputField(
        desc="the exploration steps successfully executed so far",
    )
    negative_feedback: Dict[str, List[ReasoningStepEntry]] = dspy.InputField(
        desc="prohibited next steps",
    )
    verbose_plan: str = dspy.OutputField(
        prefix="Result or next exploration step:",
        desc=(
            "The action objective has been fulfilled, the result is ... "
            "OR verbose describe the next step plan with IP address and exact command "
            "OR Finish because command execution failed"
        )
    )


class NextStepPredictor(dspy.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.adaptive_planning = dspy.ChainOfThought(NextStepPlanning, max_retries=5)

    def forward(self, context, query, acquired_information, explored_paths, negative_feedback):
        logger.debug(f"Running {self.__class__.__name__}")
        with dspy.context(lm=self.lm["gpt4o_mini"]):
            exploration_plan = self.adaptive_planning(
                context=context,
                objective=query,
                acquired_information=acquired_information,
                explored_reasoning_steps=explored_paths,
                negative_feedback=negative_feedback,
            )
        return exploration_plan.verbose_plan, exploration_plan.reasoning


class ConsecutiveCounter:
    def __init__(self, threshold: int = 2):
        self.threshold = threshold
        self._count = 0

    def increment(self) -> bool:
        self._count += 1
        if self._count >= self.threshold:
            self._count = 0
            return True
        return False

    def get(self) -> int:
        return self._count

    def reset(self):
        self._count = 0


class ShowCommandSynthesis(BaseModel):
    lm: dict = None
    _context: Optional[List] = None
    _netio_cli: DeviceCommandLineAPI = PrivateAttr()
    _loop_counter = ConsecutiveCounter()

    def __init__(self, **data):
        super().__init__(**data)
        self._netio_cli = DeviceCommandLineAPI(lm=self.lm)

    def _exploration_output_has_data(self, exploration_output) -> bool:
        for result_list in exploration_output.values():
            if not isinstance(result_list, list):
                continue
            for result in result_list:
                node_ip = result.get('node_ip_address', '')
                cmd_output = result.get('command_output', [])
                if node_ip and cmd_output:
                    return True
        return False

    def _build_negative_feedback_cache(
        self,
        negative_feedback: Dict[str, List[ReasoningStepEntry]],
        expiry_minutes: int = 10,
        skip_statuses: Set[str] = None,
        cleanup_storage: bool = True
    ) -> Tuple[Dict[str, Set[str]], Dict[str, List[ReasoningStepEntry]]]:
        """
        Build an optimized cache for O(1) negative feedback lookups with time-based expiration.
        Also cleans up expired entries from storage if cleanup_storage is True.

        Args:
            negative_feedback: Current negative feedback from ShortTermMem
            expiry_minutes: Minutes after which negative feedback expires (default: 10)
            skip_statuses: Set of statuses to consider for skipping (default: {"failed"})
            cleanup_storage: Whether to clean expired entries from storage (default: True)

        Returns:
            Tuple of (cache_dict, cleaned_storage_dict)
            - cache_dict: Dict mapping IP addresses to sets of failed commands
            - cleaned_storage_dict: Storage data with expired entries removed
        """
        if skip_statuses is None:
            skip_statuses = {"failed"}

        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=expiry_minutes)
        cache = {}
        cleaned_storage = {}
        expired_count = 0
        cleaned_count = 0

        for ip_address, entries in negative_feedback.items():
            failed_commands = set()
            active_entries = []

            # Deduplicate entries by command (keep most recent)
            command_entries = {}
            for entry in entries:
                try:
                    if entry.command not in command_entries or entry.timestamp > command_entries[entry.command].timestamp:
                        command_entries[entry.command] = entry
                except (AttributeError, TypeError) as e:
                    netio_logger.warning(f"Invalid entry in negative feedback for {ip_address}: {e}")
                    continue

            # Process deduplicated entries
            for command, entry in command_entries.items():
                if entry.timestamp > cutoff_time:
                    active_entries.append(entry)
                    if entry.status in skip_statuses:
                        failed_commands.add(entry.command)
                else:
                    expired_count += 1
                    if cleanup_storage:
                        cleaned_count += 1

            # Only keep IPs that have active entries
            if active_entries:
                cleaned_storage[ip_address] = active_entries
                if failed_commands:
                    cache[ip_address] = failed_commands

        if expired_count > 0:
            netio_logger.info(f" ⏰ Expired {expired_count} negative feedback entries older than {expiry_minutes}min")

        if cleanup_storage and cleaned_count > 0:
            netio_logger.info(f" 🧹 Cleaned {cleaned_count} expired entries from negative feedback storage")

        return cache, cleaned_storage

    def _update_negative_feedback_entry(
        self,
        ip_address: str,
        command: str,
        status: str,
        info: str,
        device_key: str
    ) -> None:
        """
        Update or remove a specific negative feedback entry.

        Args:
            ip_address: IP address of the device
            command: Command that was executed
            status: Status of execution ("success" or "failed")
            info: Additional information about the execution
            device_key: Composite key of device IP and command
        """
        feedback_entry = ReasoningStepEntry(
            command=command,
            device_key=device_key,
            status=status,
            info=info,
            timestamp=datetime.now(timezone.utc).replace(microsecond=0)
        )

        def update_feedback(neg_fb):
            d = neg_fb or {}
            if status == "success":
                # Remove the command from negative feedback on success
                if ip_address in d:
                    command_exists = any(entry.command == command for entry in d[ip_address])
                    if command_exists:
                        d[ip_address] = [entry for entry in d[ip_address] if entry.command != command]
                        netio_logger.info(f" ✅ Removed successful command '{command}' from negative feedback for {ip_address}")
                        if not d[ip_address]:
                            del d[ip_address]
            else:
                # Update or add failed command entry
                if ip_address not in d:
                    d[ip_address] = []
                d[ip_address] = [entry for entry in d[ip_address] if entry.command != command]
                d[ip_address].append(feedback_entry)
                netio_logger.info(f" ❌ Updated negative feedback for '{command}' on {ip_address}")

            return d

        ShortTermMem.update("negative_feedback", update_feedback, default={})

    def _should_skip_command(
        self,
        device: DeviceCommand,
        negative_cache: Dict[str, Set[str]]
    ) -> Tuple[bool, str]:
        """
        Determine if a command should be skipped based on negative feedback.

        Args:
            device: The device command to check
            negative_cache: Pre-built cache of failed commands by IP

        Returns:
            Tuple of (should_skip: bool, reason: str)
        """
        ip_str = str(device.ip_address)

        if ip_str not in negative_cache:
            return False, ""

        if device.command in negative_cache[ip_str]:
            reason = f"Command '{device.command}' previously failed on {ip_str}"
            return True, reason

        return False, ""

    def _log_filtering_metrics(
        self,
        original_count: int,
        filtered_count: int,
        skipped_count: int,
        processing_time_ms: float
    ) -> None:
        """
        Log comprehensive metrics for monitoring and debugging.

        Args:
            original_count: Number of commands before filtering
            filtered_count: Number of commands after filtering
            skipped_count: Number of commands skipped
            processing_time_ms: Time taken for filtering in milliseconds
        """
        if skipped_count > 0:
            skip_percentage = (skipped_count / original_count) * 100 if original_count > 0 else 0
            netio_logger.info(
                f" 📊 Filtering metrics: {original_count} → {filtered_count} commands "
                f"({skipped_count} skipped, {skip_percentage:.1f}%) in {processing_time_ms:.2f}ms"
            )
        else:
            netio_logger.debug(
                f" 📊 Filtering metrics: {original_count} commands processed, "
                f"none skipped in {processing_time_ms:.2f}ms"
            )

    def _filter_exploration_list(
        self,
        exploration_list: List[DeviceCommand],
        expiry_minutes: int = 10
    ) -> List[DeviceCommand]:
        """
        Filter exploration list to remove commands that should be skipped based on negative feedback.

        Args:
            exploration_list: List of commands to potentially execute
            expiry_minutes: Minutes after which negative feedback expires

        Returns:
            Filtered list with problematic commands removed
        """
        start_time = datetime.now()

        if not exploration_list:
            return exploration_list

        original_count = len(exploration_list)

        negative_feedback = ShortTermMem.get("negative_feedback", {})
        if not negative_feedback:
            netio_logger.debug("No negative feedback found, proceeding with all commands")
            return exploration_list

        negative_cache, cleaned_storage = self._build_negative_feedback_cache(negative_feedback, expiry_minutes)

        # Update storage with cleaned data if cleanup occurred
        if cleaned_storage != negative_feedback:
            ShortTermMem.set("negative_feedback", cleaned_storage)
            netio_logger.debug("Updated negative feedback storage with cleaned data")

        if not negative_cache:
            netio_logger.debug("No active failed commands in negative feedback, proceeding with all commands")
            return exploration_list

        filtered_list = []
        skipped_count = 0

        for device in exploration_list:
            should_skip, reason = self._should_skip_command(device, negative_cache)

            if should_skip:
                skipped_count += 1
                netio_logger.info(f" ⏭️  Skipping command: {reason}")
            else:
                filtered_list.append(device)
                netio_logger.info(f" ✅ Allowing command '{device.command}' on {device.ip_address}")

        processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        self._log_filtering_metrics(original_count, len(filtered_list), skipped_count, processing_time_ms)

        return filtered_list

    def _contains_ip(self, text: str) -> bool:
        ip_candidate_pattern = re.compile(r'\b(?:[0-9]{1,3}(?:\.[0-9]{1,3}){3}|[A-Fa-f0-9:]+)\b')
        for match in ip_candidate_pattern.findall(text):
            try:
                ipaddress.ip_address(match)  # IPv4 or IPv6
                return True
            except ValueError:
                continue
        return False

    def run(self, query: str) -> tuple[str, str]:
        logger.debug(f"Running {self.__class__.__name__}")

        current_plan = query or ""
        context = ShortTermMem.get("high_impact") or self._context or []
        ShortTermMem.update("explored_reasoning_steps", lambda x: x or {}, default={})
        ShortTermMem.update("explored_reasoning_outputs", lambda x: x or {}, default={})
        ShortTermMem.update("negative_feedback", lambda x: x or {}, default={})

        # Network exploration
        exploration_output = {}
        if not self._contains_ip(current_plan):

            # Handling termination
            if self._loop_counter.increment():
                msg1 = "Finish all exploration commands."
                msg2 = (
                    f"Exploration failed {self._loop_counter.get()} times because we were unable "
                    f"to identify available network devices by IP address and correct syntax command. "
                )
                current_plan = "Finish the exploration"
                ShortTermMem.set("user_query", current_plan)
                return msg1, msg2
            else:
                self._loop_counter.reset()
                msg1 = (
                    f"Please provide a plan matching your intent {current_plan} and with "
                    f"corresponding device IP address and a correct syntax command."
                )
                msg2 = "Exploration plan does not contain an IP address and correct syntax command. "
                current_plan = (
                    f"Find a correct syntax command for my intent {current_plan} and "
                    f"then try again."
                )
                ShortTermMem.set("user_query", current_plan)
                netio_logger.warning(f" ❌ No IP address in plan: {current_plan}")
                return msg1, msg2
        else:
            netio_logger.info(f"Exploration intent: {current_plan}")

            # Generate commands
            exploration_list = self._netio_cli._show_command_pipeline(current_plan)

            # Filter based on negative feedback
            filtered_exploration_list = self._filter_exploration_list(exploration_list)

            # Check if all commands were filtered out
            if not filtered_exploration_list:
                netio_logger.warning(" ⚠️  All commands in the exploration plan failed.")
                verbose_plan = "All proposed commands have previously failed. No new exploration possible with current plan."
                reasoning = "All commands in the exploration plan have negative feedback indicating previous failures."
                return verbose_plan, reasoning

            # Execute filtered commands
            for device in filtered_exploration_list:
                device_output = self._netio_cli._command_execution_pipeline(device)
                device_key = f"{str(device.ip_address)}:{device.command}"
                if isinstance(device_output, str):
                    try:
                        exploration_output[device_key] = json.loads(device_output)
                    except Exception as e:
                        netio_logger.debug(f" ❌ Could not json parse output for {device_key}: {e}")
                        exploration_output[device_key] = device_output  # fallback
                else:
                    exploration_output[device_key] = device_output

                # Exploration command rewards
                if not self._exploration_output_has_data(exploration_output):
                    netio_logger.warning(f" ❌ No data found in exploration output for {device_key}.")
                    exploration_output[device_key] = [{
                        "node_ip_address": str(device.ip_address),
                        "command_output": [
                            "No data was received from this node IP address.",
                            "You likely do not have permission to execute commands on this device."
                            "Do not retry.",
                        ],
                    }]

                    # Update negative feedback using the new helper method
                    ip_key = str(device.ip_address)
                    self._update_negative_feedback_entry(
                        ip_key,
                        device.command,
                        "failed",
                        "No data received from this device. Do not retry. Likely incorrect command syntax.",
                        device_key
                    )
                    netio_logger.info(f" ❌ Skip exploration step (zero reward): {device_key}")
                else:
                    # Command succeeded - remove from negative feedback and add to positive tracking
                    ip_key = str(device.ip_address)
                    self._update_negative_feedback_entry(
                        ip_key,
                        device.command,
                        "success",
                        "Command executed successfully and data was received.",
                        device_key
                    )

                    feedback_entry = ReasoningStepEntry(
                        command=device.command,
                        device_key=device_key,
                        status="success",
                        info="Command executed successfully and data was received.",
                        timestamp=datetime.now(timezone.utc).replace(microsecond=0)
                    )

                    # Update simple policy models
                    def add_reasoning_step(steps):
                        d = steps or {}
                        d.setdefault(ip_key, []).append(feedback_entry)
                        return d

                    ShortTermMem.update("explored_reasoning_steps", add_reasoning_step, default={})

                    def add_reasoning_output(outputs):
                        d = outputs or {}
                        d[device_key] = exploration_output[device_key]
                        return d

                    ShortTermMem.update("explored_reasoning_outputs", add_reasoning_output, default={})

            solution_state, reasoning = NextStepPredictor(lm=self.lm)(
                context=context,
                query=current_plan,
                acquired_information=ShortTermMem.get("explored_reasoning_outputs"),
                explored_paths=ShortTermMem.get("explored_reasoning_steps"),
                negative_feedback=ShortTermMem.get("negative_feedback"),
            )
            netio_logger.info(f" 👣 Next step: '{solution_state}'")
            return solution_state, reasoning
