#!/usr/local/bin/python3
"""
NOC MCP Server
A Model Context Protocol server for Network Operations Center tools using DSPY framework.
Supports both FastMCP (interactive) and stdio (MCP Inspector) modes.
"""

import asyncio
import logging
import argparse
import sys
import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add parent directory to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp.server.fastmcp import FastMCP
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolRequest
import mcp.types as types
from core.tool_registry import ToolRegistry
from core.mcp_tools import MCPToolWrapper
from config import Settings

# Set up logging to stderr to avoid protocol interference
logging.basicConfig(
    level=logging.ERROR,  # Only show errors to avoid protocol interference
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

logger = logging.getLogger(__name__)


class MinimalMCPServer:
    """Minimal MCP server implementation for inspector compatibility."""
    
    def __init__(self):
        # Set environment variables
        os.environ.setdefault('OPENAI_DEFAULT_API_KEY', 'demo-key')
        os.environ.setdefault('OPENAI_DEFAULT_MODEL', 'openai/gpt-4o-mini')
        
        # Initialize components
        self.tool_registry = ToolRegistry()
        self.tools = {}
        self.tool_wrappers = {}
        self.setup_tools()
    
    def setup_tools(self):
        """Setup tool wrappers."""
        tools = self.tool_registry.get_tools_for_package("syslog")
        
        for tool_name in tools:
            try:
                wrapper = MCPToolWrapper.create_tool(tool_name)
                if wrapper:
                    self.tool_wrappers[tool_name] = wrapper
                    self.tools[tool_name.lower()] = {
                        "name": tool_name.lower(),
                        "description": wrapper.get_description(),
                        "inputSchema": wrapper.get_parameters_schema()
                    }
            except Exception as e:
                logger.error(f"Error setting up tool {tool_name}: {e}")
    
    async def handle_request(self, request):
        """Handle MCP requests."""
        try:
            method = request.get("method")
            request_id = request.get("id")
            
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "NOC MCP Server",
                            "version": "1.0.0"
                        }
                    }
                }
            
            elif method == "initialized":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {}
                }
            
            elif method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": list(self.tools.values())
                    }
                }
            
            elif method == "tools/call":
                params = request.get("params", {})
                tool_name = params.get("name")
                arguments = params.get("arguments", {})
                
                # Find wrapper
                wrapper_key = None
                for key in self.tool_wrappers:
                    if key.lower() == tool_name:
                        wrapper_key = key
                        break
                
                if not wrapper_key:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Tool '{tool_name}' not found"
                        }
                    }
                
                try:
                    wrapper = self.tool_wrappers[wrapper_key]
                    result = await wrapper.execute(arguments)
                    
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": str(result)
                                }
                            ]
                        }
                    }
                except Exception as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32000,
                            "message": f"Tool execution error: {str(e)}"
                        }
                    }
            
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown method: {method}"
                    }
                }
                
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32000,
                    "message": f"Server error: {str(e)}"
                }
            }
    
    async def run_stdio(self):
        """Run the server in stdio mode."""
        logger.error("Starting minimal MCP server...")
        
        try:
            while True:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    request = json.loads(line)
                    response = await self.handle_request(request)
                    print(json.dumps(response), flush=True)
                except json.JSONDecodeError as e:
                    error_response = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {
                            "code": -32700,
                            "message": f"Parse error: {str(e)}"
                        }
                    }
                    print(json.dumps(error_response), flush=True)
                except Exception as e:
                    logger.error(f"Request handling error: {e}")
                    
        except Exception as e:
            logger.error(f"Server error: {e}")


class NOCMCPServer:
    """
    NOC MCP Server with DSPY integration for network operations tools.
    """
    
    def __init__(self, port: int = 8075, tool_package: str = "syslog"):
        """
        Initialize NOC MCP Server.
        
        Args:
            port: Port to run the server on
            tool_package: Tool package to load from config/tool_packages.yaml
        """
        self.port = port
        self.tool_package = tool_package
        self.mcp = FastMCP("NOC MCP Server", "1.0.0", port=port)
        self.tool_registry = ToolRegistry()
        self.settings = Settings()
        self.tools = {}
        self.tool_wrappers = {}
        
        logger.info(f"Initializing NOC MCP Server on port {port} with tool package '{tool_package}'")
        
    def setup_tools(self):
        """Setup MCP tools based on the selected tool package."""
        logger.info(f"Setting up tools for package: {self.tool_package}")
        
        # Get tools from registry based on package
        tools = self.tool_registry.get_tools_for_package(self.tool_package)
        
        for tool_name in tools:
            try:
                # Create MCP tool wrapper
                tool_wrapper = MCPToolWrapper.create_tool(tool_name)
                if tool_wrapper:
                    # Register tool with FastMCP
                    self.register_mcp_tool(tool_name, tool_wrapper)
                    logger.info(f"Registered tool: {tool_name}")
                else:
                    logger.warning(f"Failed to create wrapper for tool: {tool_name}")
                    
            except Exception as e:
                logger.error(f"Error setting up tool {tool_name}: {e}")
                
    def register_mcp_tool(self, tool_name: str, tool_wrapper: 'MCPToolWrapper'):
        """Register a tool with the MCP server."""
        
        # Create the tool function dynamically
        async def tool_func(**kwargs) -> str:
            """Dynamic MCP tool."""
            try:
                result = await tool_wrapper.execute(kwargs)
                if isinstance(result, dict):
                    import json
                    return json.dumps(result, indent=2)
                return str(result)
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                return f"Error: {str(e)}"
        
        # Set function metadata for FastMCP
        tool_func.__name__ = tool_name.lower()
        tool_func.__doc__ = tool_wrapper.get_description()
        
        # Register with FastMCP - simplified approach
        decorated_func = self.mcp.tool(
            description=tool_wrapper.get_description()
        )(tool_func)
        
        # Store reference to prevent garbage collection
        setattr(self, f"_tool_{tool_name.lower()}", decorated_func)
        
    def setup_server_tools(self):
        """Setup server management tools synchronously."""
        # Add health check tool
        @self.mcp.tool(description="Check server health status")
        async def health_check() -> str:
            """Check server health status."""
            return "✅ NOC MCP Server is running and healthy"
            
        # Add list tools functionality
        @self.mcp.tool(description="List all available tools in the current package")
        async def list_available_tools() -> str:
            """List all available tools in the current package."""
            tools = self.tool_registry.get_tools_for_package(self.tool_package)
            tool_list = "\n".join([f"- {tool}" for tool in tools])
            return f"Available tools in '{self.tool_package}' package:\n{tool_list}"
        
        # Store references to prevent garbage collection
        self._health_check = health_check
        self._list_available_tools = list_available_tools
        
    def _get_tool_schema(self, tool_wrapper: 'MCPToolWrapper') -> dict:
        """Get tool schema for MCP registration."""
        return {
            "name": tool_wrapper.name,
            "description": tool_wrapper.get_description(),
            "parameters": tool_wrapper.get_parameters_schema()
        }
        
    async def start(self):
        """Start the FastMCP server."""
        try:
            logger.info(f"Starting NOC MCP Server on port {self.port}")
            logger.info(f"Package: {self.tool_package}")
            
            # Setup tools
            self.setup_tools()
            self.setup_server_tools()
            
            logger.info(f"Tools available: {len(self.tools)}")
            
            # Run the FastMCP server
            await self.mcp.run()
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="NOC MCP Server")
    parser.add_argument("--port", type=int, default=8075,
                        help="Port to run the server on (default: 8075)")
    parser.add_argument("--tool-package", type=str, default="syslog",
                        help="Tool package to load (default: syslog)")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Log file path (optional)")
    return parser.parse_args()


def setup_logging(log_level: str, log_file: Optional[str] = None):
    """Setup logging configuration with optional file output."""
    # Clear any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set logging level
    root_logger.setLevel(getattr(logging, log_level))


async def main():
    """Main function for stdio mode (MCP Inspector)."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Create and run minimal MCP server
    server = MinimalMCPServer()
    await server.run_stdio()


def main_sync():
    """Synchronous main function for FastMCP compatibility."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Check if running in stdio mode (for MCP Inspector)
    if not sys.stdin.isatty():
        # Stdio mode - run minimal server for MCP Inspector
        logger.info("Starting NOC MCP Server in stdio mode for MCP Inspector...")
        asyncio.run(main())
    else:
        # Interactive mode - use FastMCP server
        logger.info("Starting NOC MCP Server in interactive mode...")
        
        # Create server
        server = NOCMCPServer(port=args.port, tool_package=args.tool_package)
        
        # Setup tools synchronously
        server.setup_tools()
        server.setup_server_tools()
        
        logger.info(f"NOC MCP Server configured on port {args.port} with tool package '{args.tool_package}'")
        
        # Start server (FastMCP handles the event loop)
        server.mcp.run()


if __name__ == "__main__":
    main_sync()
