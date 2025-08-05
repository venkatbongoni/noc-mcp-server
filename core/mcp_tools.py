"""
MCP Tool Wrappers for NOC Tools
Provides DSPY-based MCP tool wrappers for existing NOC tools.
"""

import logging
import asyncio
from typing import Any, Dict, Optional, Type
from abc import ABC, abstractmethod

# Import existing tools
from tools.log_message_reader import LogMessageReader
from tools.syslog_high_impact_evaluator import DecideHighestImpactIssue
from tools.show_commands_synthesis import ShowCommandSynthesis
from tools.syslog_holistic_reasoner import SyslogHolisticReasoner
from tools.objective_analyzer import AnalyzeTheObjectiveFulfillment
from lib.datastore import ShortTermMem
from config import Settings

logger = logging.getLogger(__name__)


class MCPToolWrapper(ABC):
    """
    Abstract base class for MCP tool wrappers.
    """
    
    def __init__(self, tool_name: str, description: str):
        self.tool_name = tool_name
        self.description = description
        self.settings = Settings()
        
    @abstractmethod
    async def execute(self, kwargs: Dict[str, Any]) -> str:
        """Execute the wrapped tool with MCP arguments."""
        pass
        
    def get_description(self) -> str:
        """Get tool description."""
        return self.description
        
    def get_parameters_schema(self) -> dict:
        """Get tool parameters schema."""
        # Default schema - can be overridden by subclasses
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Query parameter"}
            }
        }
        
    @staticmethod
    def create_tool(tool_name: str) -> Optional['MCPToolWrapper']:
        """
        Factory method to create tool wrappers.
        
        Args:
            tool_name: Name of the tool to create
            
        Returns:
            MCPToolWrapper instance or None if tool not found
        """
        tool_map = {
            "READ_NEW_LOG_MESSAGES": ReadNewLogMessagesWrapper,
            "HIGHEST_SERVICE_IMPACT": HighestServiceImpactWrapper,
            "QUERY_DEVICES": QueryDevicesWrapper,
            "HOLISTIC_CAUSAL_REASONING": HolisticCausalReasoningWrapper,
            "EVALUATE_DRAFT_ANSWER": EvaluateDraftAnswerWrapper,
            "SHORT_TERM_MEMORY": ShortTermMemoryWrapper,
        }
        
        tool_class = tool_map.get(tool_name)
        if tool_class:
            return tool_class()
        else:
            logger.warning(f"Unknown tool: {tool_name}")
            return None


class ReadNewLogMessagesWrapper(MCPToolWrapper):
    """MCP wrapper for log message reading tool."""
    
    def __init__(self):
        super().__init__(
            "READ_NEW_LOG_MESSAGES",
            "Read the most recent syslog data from the network"
        )
        # Initialize with default LM config
        self.lm = self._get_default_lm_config()
        
    def _get_default_lm_config(self) -> Dict[str, Any]:
        """Get default language model configuration."""
        # This would typically come from settings
        return {
            'gpt4o_mini': None  # Placeholder for actual LM configuration
        }
        
    async def execute(self, kwargs: Dict[str, Any]) -> str:
        """Execute log message reading."""
        try:
            # Extract parameters from MCP kwargs
            query = kwargs.get("query", "Analyze recent log messages")
            limit = kwargs.get("limit", 100)
            
            tool = LogMessageReader(lm=self.lm)
            result = tool.run(query)
            return f"Log analysis results (limit: {limit}):\n{result}"
        except Exception as e:
            logger.error(f"Error in ReadNewLogMessagesWrapper: {e}")
            return f"Error reading log messages: {str(e)}"


class HighestServiceImpactWrapper(MCPToolWrapper):
    """MCP wrapper for service impact evaluation tool."""
    
    def __init__(self):
        super().__init__(
            "HIGHEST_SERVICE_IMPACT",
            "Perform targeted operational impact analysis to identify the severity and scope of the problem"
        )
        self.lm = self._get_default_lm_config()
        
    def _get_default_lm_config(self) -> Dict[str, Any]:
        """Get default language model configuration."""
        return {
            'gpt4o_mini': None
        }
        
    async def execute(self, kwargs: Dict[str, Any]) -> str:
        """Execute service impact evaluation."""
        try:
            query = kwargs.get("query", "Evaluate service impact")
            timeframe = kwargs.get("timeframe", "1h")
            
            tool = DecideHighestImpactIssue(lm=self.lm)
            result = tool.run(query)
            return f"Service impact analysis (timeframe: {timeframe}):\n{result}"
        except Exception as e:
            logger.error(f"Error in HighestServiceImpactWrapper: {e}")
            return f"Error evaluating service impact: {str(e)}"


class QueryDevicesWrapper(MCPToolWrapper):
    """MCP wrapper for device query tool."""
    
    def __init__(self):
        super().__init__(
            "QUERY_DEVICES", 
            "Real time execute commands on network nodes to gather diagnostic data and perform network exploration"
        )
        self.lm = self._get_default_lm_config()
        
    def _get_default_lm_config(self) -> Dict[str, Any]:
        """Get default language model configuration."""
        return {
            'gpt4o_mini': None
        }
        
    async def execute(self, kwargs: Dict[str, Any]) -> str:
        """Execute device queries."""
        try:
            query = kwargs.get("query", "Query network devices")
            device_type = kwargs.get("device_type", "router")
            command = kwargs.get("command", "show version")
            
            tool = ShowCommandSynthesis(lm=self.lm)
            result = tool.run(f"{query} - Device: {device_type}, Command: {command}")
            return f"Device query results (type: {device_type}):\n{result}"
        except Exception as e:
            logger.error(f"Error in QueryDevicesWrapper: {e}")
            return f"Error querying devices: {str(e)}"


class HolisticCausalReasoningWrapper(MCPToolWrapper):
    """MCP wrapper for holistic causal reasoning tool."""
    
    def __init__(self):
        super().__init__(
            "HOLISTIC_CAUSAL_REASONING",
            "Conduct a comprehensive analysis of collected data to determine the root cause of a network issue"
        )
        self.lm = self._get_default_lm_config()
        
    def _get_default_lm_config(self) -> Dict[str, Any]:
        """Get default language model configuration."""
        return {
            'gpt4o_mini': None
        }
        
    async def execute(self, kwargs: Dict[str, Any]) -> str:
        """Execute holistic causal reasoning."""
        try:
            query = kwargs.get("query", "Perform root cause analysis")
            data_sources = kwargs.get("data_sources", [])
            
            tool = SyslogHolisticReasoner(lm=self.lm)
            result = tool.run(query)
            return f"Holistic causal reasoning (data sources: {data_sources}):\n{result}"
        except Exception as e:
            logger.error(f"Error in HolisticCausalReasoningWrapper: {e}")
            return f"Error in causal reasoning: {str(e)}"


class EvaluateDraftAnswerWrapper(MCPToolWrapper):
    """MCP wrapper for draft answer evaluation tool."""
    
    def __init__(self):
        super().__init__(
            "EVALUATE_DRAFT_ANSWER",
            "Evaluate whether the proposed answer resolves the users question or if additional data exploration is required"
        )
        self.lm = self._get_default_lm_config()
        
    def _get_default_lm_config(self) -> Dict[str, Any]:
        """Get default language model configuration."""
        return {
            'gpt4o_mini': None
        }
        
    async def execute(self, kwargs: Dict[str, Any]) -> str:
        """Execute draft answer evaluation."""
        try:
            draft_answer = kwargs.get("draft_answer", "")
            user_question = kwargs.get("user_question", "")
            query = kwargs.get("query", f"Evaluate draft answer: {draft_answer}")
            
            tool = AnalyzeTheObjectiveFulfillment(lm=self.lm)
            result = tool.run(query)
            return f"Draft answer evaluation:\nQuestion: {user_question}\nAnswer: {draft_answer}\nEvaluation: {result}"
        except Exception as e:
            logger.error(f"Error in EvaluateDraftAnswerWrapper: {e}")
            return f"Error evaluating draft answer: {str(e)}"


class ShortTermMemoryWrapper(MCPToolWrapper):
    """MCP wrapper for short term memory operations."""
    
    def __init__(self):
        super().__init__(
            "SHORT_TERM_MEMORY",
            "Store and retrieve information from short term memory"
        )
        
    async def execute(self, kwargs: Dict[str, Any]) -> str:
        """Execute memory operations."""
        try:
            action = kwargs.get("action", "get")
            key = kwargs.get("key", "context")
            value = kwargs.get("value")
            
            if action == "store" and value is not None:
                ShortTermMem.set(key, value)
                return f"Stored '{key}' = '{value}' in short-term memory"
            elif action == "retrieve":
                result = ShortTermMem.get(key, "No data found")
                return f"Retrieved '{key}' from memory: {result}"
            elif action == "list":
                # Mock list of memory contents
                return "Short-term memory contents:\n- network_issue_id: INC-2025-001\n- analysis_status: in_progress\n- primary_device: router-core-01"
            else:
                return f"Unknown memory action: {action}. Supported: store, retrieve, list"
        except Exception as e:
            logger.error(f"Error in ShortTermMemoryWrapper: {e}")
            return f"Error with memory operation: {str(e)}"
