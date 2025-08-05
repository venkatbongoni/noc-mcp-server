"""
NOC MCP Client
Client for connecting to the NOC MCP Server and using its tools.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)


class NOCMCPClient:
    """
    Client for connecting to and using NOC MCP Server tools.
    Usage:
        async with NOCMCPClient(server_url) as client:
            tools = client.get_available_tools()
            result = await client.call_tool("tool_name", arg1="value")
    """
    
    def __init__(self, server_url: str = "http://localhost:8075", timeout: float = 30.0):
        """
        Initialize NOC MCP Client.
        
        Args:
            server_url: URL of the NOC MCP Server
            timeout: Timeout for requests
        """
        self.server_url = server_url
        self.timeout = timeout
        self.session: Optional[ClientSession] = None
        self.available_tools: List[Dict[str, Any]] = []
        self._session_context = None
        
        logger.info(f"Initializing NOC MCP Client for server: {server_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
        
    async def connect(self):
        """Connect to the NOC MCP Server using proper async context manager."""
        try:
            logger.info(f"Connecting to NOC MCP Server at {self.server_url}")
            
            # Create and enter the SSE client context
            self._session_context = sse_client(self.server_url)
            self.session = await self._session_context.__aenter__()
            
            # Discover available tools
            await self.discover_tools()
            
            logger.info(f"Connected to NOC MCP Server. Available tools: {len(self.available_tools)}")
            
        except Exception as e:
            logger.error(f"Failed to connect to NOC MCP Server: {e}")
            raise
            
    async def disconnect(self):
        """Disconnect from the NOC MCP Server."""
        try:
            if self._session_context and self.session:
                # Exit the async context manager
                await self._session_context.__aexit__(None, None, None)
                logger.info("Disconnected from NOC MCP Server")
            elif self.session:
                # Fallback for direct session cleanup
                await self.session.close()
                logger.info("Disconnected from NOC MCP Server")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
        finally:
            self.session = None
            self._session_context = None
                
    async def discover_tools(self):
        """Discover available tools from the server."""
        if not self.session:
            logger.warning("No active session, skipping tool discovery")
            return
            
        try:
            # List available tools
            result = await self.session.list_tools()
            self.available_tools = result.tools
            
            logger.info(f"Discovered {len(self.available_tools)} tools")
            for tool in self.available_tools:
                logger.debug(f"Tool: {tool.name} - {tool.description}")
                
        except Exception as e:
            logger.warning(f"Could not discover tools (this may be expected): {e}")
            # Set empty tools list as fallback
            self.available_tools = []
            
    async def call_tool(self, tool_name: str, **kwargs) -> str:
        """
        Call a tool on the server.
        
        Args:
            tool_name: Name of the tool to call
            **kwargs: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        if not self.session:
            raise RuntimeError("Not connected to server")
            
        try:
            logger.debug(f"Calling tool '{tool_name}' with args: {kwargs}")
            
            # Call the tool
            result = await self.session.call_tool(tool_name, kwargs)
            
            logger.debug(f"Tool '{tool_name}' returned: {result}")
            return str(result.content)
            
        except Exception as e:
            logger.error(f"Error calling tool '{tool_name}': {e}")
            raise
            
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [tool.name for tool in self.available_tools]
        
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific tool."""
        for tool in self.available_tools:
            if tool.name == tool_name:
                return {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
        return None
        
    async def health_check(self) -> bool:
        """Check if the server is healthy."""
        try:
            result = await self.call_tool("health_check")
            return "running" in result.lower()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
            
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


# Example usage functions
async def example_syslog_analysis():
    """Example of using NOC MCP Client for syslog analysis."""
    async with NOCMCPClient() as client:
        logger.info("Starting syslog analysis example")
        
        # Check available tools
        tools = client.get_available_tools()
        logger.info(f"Available tools: {tools}")
        
        # Read log messages
        if "read_new_log_messages" in tools:
            log_result = await client.call_tool(
                "read_new_log_messages",
                query="Analyze recent network errors"
            )
            logger.info(f"Log analysis result: {log_result}")
            
        # Evaluate service impact
        if "highest_service_impact" in tools:
            impact_result = await client.call_tool(
                "highest_service_impact",
                query="Evaluate impact of recent errors"
            )
            logger.info(f"Impact analysis result: {impact_result}")
            
        # Perform root cause analysis
        if "holistic_causal_reasoning" in tools:
            rca_result = await client.call_tool(
                "holistic_causal_reasoning",
                query="Determine root cause of network issues"
            )
            logger.info(f"Root cause analysis result: {rca_result}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(example_syslog_analysis())
