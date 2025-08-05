# -----------------------------------------------------------------------------
# Copyright (c) 2024-2025 Cisco Systems.
# Author: Timo Koehler
#
# Licensed under the MIT License. See LICENSE file in the project root.
#
# Portions of this software are protected by U.S. Patent No. [Patent Number],
# held by Cisco Systems.
# -----------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import List, Optional
from lib import ModeType
from .signatures import Tool
from config import Settings

# Tools
from tools.syslog_holistic_reasoner import SyslogHolisticReasoner
from tools.log_message_reader import LogMessageReader
from tools.syslog_high_impact_evaluator import DecideHighestImpactIssue
from tools.show_commands_synthesis import ShowCommandSynthesis
from tools.objective_analyzer import AnalyzeTheObjectiveFulfillment
# from tools.splunk_rag import SplunkRag
# from tools.splunk_ics import SplunkICSAPI

logger = logging.getLogger(__name__)
netio_logger = logging.getLogger("netio")

try:
    settings = Settings()
except Exception as e:
    logger.error(f"Failed to initialize settings: {e}")
    raise


def define_tools(
    context, lm: dict, mode: ModeType = "inference",
    model_path: Optional[Path] = None,  user_query: Optional[str] = None,
    signature: Optional[str] = None
) -> List[Tool]:
    """
    Define the agent's tools.
    """
    # Extract and validate ICS configuration (disabled for now)
    # try:
    #     ics_rag_url = str(settings.ics_config.rag_url)
    #     ics_sql_url = str(settings.ics_config.sql_url)

    #     if not ics_rag_url or not ics_sql_url:
    #         raise ValueError("ICS URLs cannot be empty")

    # except Exception as e:
    #     logger.error(f"ICS configuration error: {e}")
    #     raise ValueError(f"Invalid ICS configuration: {e}")

    if mode == "inference" or mode == "optimization":
        tools = [
            Tool(
                tool=LogMessageReader(lm=lm),
                name="READ_NEW_LOG_MESSAGES",
                desc="Read the most recent syslog data from the network"
            ),
            Tool(
                tool=DecideHighestImpactIssue(lm=lm),
                name="HIGHEST_SERVICE_IMPACT",
                desc=(
                    "Perform targeted operational impact analysis to identify the severity "
                    "and scope of the problem. Explanation: Specific evaluation of impact "
                    "to prioritize severity."
                ),
            ),
            Tool(
                tool=ShowCommandSynthesis(lm=lm),
                name="QUERY_DEVICES",
                desc=(
                    "Real time execute commands on network nodes to gather diagnostic "
                    "data and perform network exploration."
                ),
            ),
            Tool(
                tool=(
                    SyslogHolisticReasoner(lm=lm, model_path=model_path)
                    if model_path is not None
                    else SyslogHolisticReasoner(lm=lm)
                ),
                name="HOLISTIC_CAUSAL_REASONING",
                desc=(
                    "Conduct a comprehensive analysis of collected data to determine the "
                    "root cause of a network issue. Explanation: Analytical reasoning to "
                    "identify root causes clearly."
                ),
            ),
            Tool(
                tool=AnalyzeTheObjectiveFulfillment(lm=lm),
                name="EVALUATE_DRAFT_ANSWER",
                desc=(
                    "Evaluate whether the proposed answer resolves the users question "
                    "or if additional data exploration is required. Explanation: "
                    "Final decision checkpoint to validate the completeness of the "
                    "analysis."
                ),
            ),
            # Tool(
            #     tool=SplunkICSAPI(
            #         lm=lm,
            #         api_url=ics_sql_url,
            #         original_query=user_query or "",
            #         signature=signature or ""
            #     ),
            #     name="LOOK_UP_FROM_ICS_SQL_SYSTEM",
            #     desc=(
            #         "Use this tool to find Cisco ICS recommendations for error "
            #         "signatures like %HA-HA_WD-2-DISK_CRIT. This tool provides "
            #         "specific fix actions and recommendations for system errors."
            #     ),
            # ),
            # Tool(
            #     tool=SplunkRag(
            #         lm=lm,
            #         api_url=ics_rag_url,
            #         original_query=user_query or "",
            #         signature=signature or ""
            #     ),
            #     name="LOOK_UP_FROM_ICS_RAGSYSTEM",
            #     desc=(
            #         "Use this Retrieval-Augmented Generation (RAG) tool to find "
            #         "Cisco ICS recommendations for error signatures like "
            #         "%OS-COREHELPER-3-CORE_COPY_ERROR('%' and include "
            #         "category(OS-COREHELPER), severity(3), and "
            #         "mnemonic(CORE_COPY_ERROR)). This tool provides specific "
            #         "fix actions and recommendations for system errors."
            #     ),
            # ),
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return tools
