"""
Tool Registry for NOC MCP Server
Manages available tools and their configurations based on tool packages.
Enhanced for Python >3.12, <3.13 with modern type hints.
"""

import logging
import yaml
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for managing NOC tools and their configurations.
    Enhanced with Python 3.12+ type hints and features.
    """
    
    def __init__(self, config_path: Optional[Path] = None) -> None:
        """
        Initialize tool registry.
        
        Args:
            config_path: Path to tool_packages.yaml config file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "tool_packages.yaml"
            
        self.config_path = config_path
        self.tool_packages = self._load_tool_packages()
        
        logger.info(f"Loaded tool packages: {list(self.tool_packages.keys())}")
        
    def _load_tool_packages(self) -> dict[str, list[str]]:
        """Load tool packages configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Extract tool packages, excluding comments - using modern dict comprehension
            packages: dict[str, list[str]] = {
                key: value for key, value in config.items() 
                if isinstance(value, list)
            }
                    
            return packages
            
        except Exception as e:
            logger.error(f"Error loading tool packages config: {e}")
            return {
                "syslog": [
                    "READ_NEW_LOG_MESSAGES",
                    "HIGHEST_SERVICE_IMPACT", 
                    "QUERY_DEVICES",
                    "HOLISTIC_CAUSAL_REASONING", 
                    "EVALUATE_DRAFT_ANSWER",
                    "SHORT_TERM_MEMORY"
                ]
            }
            
    def get_available_packages(self) -> list[str]:
        """Get list of available tool packages."""
        return list(self.tool_packages.keys())
        
    def get_tools_for_package(self, package_name: str) -> list[str]:
        """
        Get list of tools for a specific package.
        
        Args:
            package_name: Name of the tool package
            
        Returns:
            List of tool names in the package
        """
        return self.tool_packages.get(package_name, [])
        
    def is_tool_in_package(self, tool_name: str, package_name: str) -> bool:
        """
        Check if a tool is part of a specific package.
        
        Args:
            tool_name: Name of the tool
            package_name: Name of the package
            
        Returns:
            True if tool is in package, False otherwise
        """
        tools = self.get_tools_for_package(package_name)
        return tool_name in tools
        
    def get_all_tools(self) -> list[str]:
        """Get list of all available tools across all packages."""
        all_tools: set[str] = set()
        for tools in self.tool_packages.values():
            all_tools.update(tools)
        return list(all_tools)
        
    def reload_config(self) -> None:
        """Reload tool packages configuration from file."""
        self.tool_packages = self._load_tool_packages()
        logger.info("Tool packages configuration reloaded")
