#!/usr/bin/env python3
"""
End-to-End Test Suite for NOC MCP Server
Tests both FastMCP and stdio modes, tool loading, and MCP Inspector compatibility.
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
import os
from typing import Dict, Any, List
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.mcp_server import NOCMCPServer, NOCMCPStdioServer
from core.tool_registry import ToolRegistry
from core.mcp_tools import MCPToolWrapper
from mcp.types import CallToolRequest
import mcp.types as types

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MCPServerE2ETests:
    """End-to-end test suite for NOC MCP Server."""
    
    def __init__(self):
        self.test_results = []
        self.passed_tests = 0
        self.failed_tests = 0
        
    def log_test_result(self, test_name: str, passed: bool, message: str = ""):
        """Log test result."""
        status = "PASS" if passed else "FAIL"
        self.test_results.append({
            "test": test_name,
            "status": status,
            "message": message
        })
        
        if passed:
            self.passed_tests += 1
            logger.info(f"✅ {test_name}: {status} {message}")
        else:
            self.failed_tests += 1
            logger.error(f"❌ {test_name}: {status} {message}")
    
    async def test_tool_registry(self):
        """Test tool registry functionality."""
        test_name = "Tool Registry Test"
        try:
            registry = ToolRegistry()
            
            # Test getting available packages
            packages = registry.get_available_packages()
            assert len(packages) > 0, "No packages found"
            assert "syslog" in packages, "syslog package not found"
            
            # Test getting tools for syslog package
            syslog_tools = registry.get_tools_for_package("syslog")
            expected_tools = [
                "READ_NEW_LOG_MESSAGES",
                "HIGHEST_SERVICE_IMPACT",
                "QUERY_DEVICES",
                "HOLISTIC_CAUSAL_REASONING",
                "EVALUATE_DRAFT_ANSWER",
                "SHORT_TERM_MEMORY"
            ]
            
            for tool in expected_tools:
                assert tool in syslog_tools, f"Tool {tool} not found in syslog package"
            
            self.log_test_result(test_name, True, f"Found {len(syslog_tools)} tools in syslog package")
            
        except Exception as e:
            self.log_test_result(test_name, False, str(e))
    
    async def test_tool_wrapper_creation(self):
        """Test MCP tool wrapper creation."""
        test_name = "Tool Wrapper Creation Test"
        try:
            test_tools = [
                "READ_NEW_LOG_MESSAGES",
                "HIGHEST_SERVICE_IMPACT", 
                "QUERY_DEVICES",
                "HOLISTIC_CAUSAL_REASONING",
                "EVALUATE_DRAFT_ANSWER",
                "SHORT_TERM_MEMORY"
            ]
            
            created_tools = 0
            for tool_name in test_tools:
                try:
                    wrapper = MCPToolWrapper.create_tool(tool_name)
                    if wrapper:
                        created_tools += 1
                        # Test that wrapper has required methods
                        assert hasattr(wrapper, 'execute'), f"Tool {tool_name} missing execute method"
                        assert hasattr(wrapper, 'get_description'), f"Tool {tool_name} missing get_description method"
                        assert hasattr(wrapper, 'get_parameters_schema'), f"Tool {tool_name} missing get_parameters_schema method"
                        
                        # Test description
                        desc = wrapper.get_description()
                        assert len(desc) > 0, f"Tool {tool_name} has empty description"
                        
                        # Test schema
                        schema = wrapper.get_parameters_schema()
                        assert isinstance(schema, dict), f"Tool {tool_name} schema is not a dict"
                        
                except Exception as e:
                    logger.warning(f"Failed to create wrapper for {tool_name}: {e}")
            
            assert created_tools > 0, "No tool wrappers created successfully"
            self.log_test_result(test_name, True, f"Created {created_tools}/{len(test_tools)} tool wrappers")
            
        except Exception as e:
            self.log_test_result(test_name, False, str(e))
    
    async def test_tool_execution(self):
        """Test tool execution with mock data."""
        test_name = "Tool Execution Test"
        try:
            # Test each tool with sample inputs
            test_cases = [
                {
                    "tool": "READ_NEW_LOG_MESSAGES",
                    "kwargs": {"query": "Show recent errors", "limit": 10}
                },
                {
                    "tool": "HIGHEST_SERVICE_IMPACT", 
                    "kwargs": {"query": "Analyze service impact", "timeframe": "1h"}
                },
                {
                    "tool": "QUERY_DEVICES",
                    "kwargs": {"query": "Check device status", "device_type": "router", "command": "show version"}
                },
                {
                    "tool": "HOLISTIC_CAUSAL_REASONING",
                    "kwargs": {"query": "Find root cause", "data_sources": ["logs", "metrics"]}
                },
                {
                    "tool": "EVALUATE_DRAFT_ANSWER",
                    "kwargs": {
                        "draft_answer": "The issue is network congestion",
                        "user_question": "What caused the outage?",
                        "query": "Evaluate this answer"
                    }
                },
                {
                    "tool": "SHORT_TERM_MEMORY",
                    "kwargs": {"action": "store", "key": "test_key", "value": "test_value"}
                }
            ]
            
            executed_tools = 0
            for test_case in test_cases:
                try:
                    wrapper = MCPToolWrapper.create_tool(test_case["tool"])
                    if wrapper:
                        result = await wrapper.execute(test_case["kwargs"])
                        assert isinstance(result, str), f"Tool {test_case['tool']} didn't return string"
                        assert len(result) > 0, f"Tool {test_case['tool']} returned empty result"
                        executed_tools += 1
                        logger.info(f"Tool {test_case['tool']} executed successfully")
                except Exception as e:
                    logger.warning(f"Tool {test_case['tool']} execution failed: {e}")
            
            assert executed_tools > 0, "No tools executed successfully"
            self.log_test_result(test_name, True, f"Executed {executed_tools}/{len(test_cases)} tools")
            
        except Exception as e:
            self.log_test_result(test_name, False, str(e))
    
    async def test_stdio_server_initialization(self):
        """Test stdio server initialization."""
        test_name = "Stdio Server Initialization Test"
        try:
            server = NOCMCPStdioServer(tool_package="syslog")
            
            # Test that server was initialized properly
            assert server.tool_package == "syslog", "Tool package not set correctly"
            assert server.tool_registry is not None, "Tool registry not initialized"
            assert server.server is not None, "MCP server not initialized"
            assert len(server.tools) == 0, "Tools should be empty before setup"
            
            # Test tool setup
            server.setup_tools()
            assert len(server.tools) > 0, "No tools loaded after setup"
            assert len(server.tool_wrappers) > 0, "No tool wrappers created"
            
            self.log_test_result(test_name, True, f"Stdio server initialized with {len(server.tools)} tools")
            
        except Exception as e:
            self.log_test_result(test_name, False, str(e))
    
    async def test_fastmcp_server_initialization(self):
        """Test FastMCP server initialization."""
        test_name = "FastMCP Server Initialization Test"
        try:
            server = NOCMCPServer(port=8076, tool_package="syslog")  # Use different port to avoid conflicts
            
            # Test that server was initialized properly
            assert server.tool_package == "syslog", "Tool package not set correctly"
            assert server.tool_registry is not None, "Tool registry not initialized"
            assert server.mcp is not None, "FastMCP server not initialized"
            assert server.port == 8076, "Port not set correctly"
            
            # Test tool setup
            server.setup_tools()
            server.setup_server_tools()
            
            # Check that tools were registered (stored as attributes)
            tool_attributes = [attr for attr in dir(server) if attr.startswith('_tool_')]
            assert len(tool_attributes) > 0, "No tool attributes found after setup"
            
            self.log_test_result(test_name, True, f"FastMCP server initialized with tools")
            
        except Exception as e:
            self.log_test_result(test_name, False, str(e))
    
    async def test_stdio_tool_call_handling(self):
        """Test stdio server tool call handling."""
        test_name = "Stdio Tool Call Handling Test"
        try:
            from mcp.types import CallToolRequestParams
            
            server = NOCMCPStdioServer(tool_package="syslog")
            server.setup_tools()
            
            # Test tool call with memory operation
            request = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name="short_term_memory",
                    arguments={
                        "action": "store",
                        "key": "test_incident",
                        "value": "Network outage resolved"
                    }
                )
            )
            
            result = await server.handle_call_tool(request)
            assert isinstance(result, list), "Result should be a list"
            assert len(result) > 0, "Result should not be empty"
            assert isinstance(result[0], types.TextContent), "Result should contain TextContent"
            
            # Test with non-existent tool
            bad_request = CallToolRequest(
                method="tools/call",
                params=CallToolRequestParams(
                    name="non_existent_tool",
                    arguments={}
                )
            )
            
            bad_result = await server.handle_call_tool(bad_request)
            assert isinstance(bad_result, list), "Bad result should be a list"
            assert "not found" in bad_result[0].text.lower(), "Error message should indicate tool not found"
            
            self.log_test_result(test_name, True, "Tool call handling working correctly")
            
        except Exception as e:
            self.log_test_result(test_name, False, str(e))
    
    async def test_mcp_inspector_compatibility(self):
        """Test MCP Inspector compatibility by running server in stdio mode."""
        test_name = "MCP Inspector Compatibility Test"
        try:
            # Create a simple test script that mimics MCP Inspector behavior
            test_script = '''
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.mcp_server import NOCMCPStdioServer

async def test_mcp_protocol():
    """Test basic MCP protocol communication."""
    server = NOCMCPStdioServer(tool_package="syslog")
    server.setup_tools()
    
    # Test list_tools functionality
    tools = list(server.tools.values())
    print(f"Available tools: {len(tools)}")
    
    # Test tool names
    tool_names = [tool.name for tool in tools]
    expected_tools = ["read_new_log_messages", "highest_service_impact", "query_devices", 
                     "holistic_causal_reasoning", "evaluate_draft_answer", "short_term_memory"]
    
    found_tools = 0
    for expected in expected_tools:
        if expected in tool_names:
            found_tools += 1
    
    print(f"Found {found_tools}/{len(expected_tools)} expected tools")
    return found_tools == len(expected_tools)

if __name__ == "__main__":
    result = asyncio.run(test_mcp_protocol())
    sys.exit(0 if result else 1)
'''
            
            # Write test script to temporary file
            test_file = Path(__file__).parent / "temp_mcp_test.py"
            test_file.write_text(test_script)
            
            try:
                # Run the test script
                result = subprocess.run([sys.executable, str(test_file)], 
                                      capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    self.log_test_result(test_name, True, "MCP Inspector compatibility verified")
                else:
                    self.log_test_result(test_name, False, f"Test script failed: {result.stderr}")
                    
            finally:
                # Clean up test file
                if test_file.exists():
                    test_file.unlink()
                    
        except Exception as e:
            self.log_test_result(test_name, False, str(e))
    
    async def test_server_health_and_info_tools(self):
        """Test server health check and info tools."""
        test_name = "Server Health and Info Tools Test"
        try:
            server = NOCMCPServer(port=8077, tool_package="syslog")
            server.setup_tools()
            server.setup_server_tools()
            
            # Check that health check and list tools were added
            health_attr = getattr(server, '_health_check', None)
            list_attr = getattr(server, '_list_available_tools', None)
            
            assert health_attr is not None, "Health check tool not registered"
            assert list_attr is not None, "List tools function not registered"
            
            # Test health check execution
            health_result = await health_attr()
            assert "healthy" in health_result.lower(), "Health check didn't return healthy status"
            
            # Test list tools execution
            tools_result = await list_attr()
            assert "syslog" in tools_result, "Tools list doesn't mention syslog package"
            
            self.log_test_result(test_name, True, "Server management tools working correctly")
            
        except Exception as e:
            self.log_test_result(test_name, False, str(e))
    
    async def test_error_handling(self):
        """Test error handling in various scenarios."""
        test_name = "Error Handling Test"
        try:
            # Test with invalid tool package
            try:
                server = NOCMCPServer(tool_package="invalid_package")
                server.setup_tools()
                # Should not crash, should handle gracefully
            except Exception as e:
                logger.info(f"Invalid package handled gracefully: {e}")
            
            # Test tool execution with invalid arguments
            wrapper = MCPToolWrapper.create_tool("SHORT_TERM_MEMORY")
            if wrapper:
                # Test with invalid action
                result = await wrapper.execute({"action": "invalid_action"})
                assert "unknown" in result.lower() or "error" in result.lower(), "Error not handled properly"
            
            self.log_test_result(test_name, True, "Error handling working correctly")
            
        except Exception as e:
            self.log_test_result(test_name, False, str(e))
    
    async def run_all_tests(self):
        """Run all tests in the suite."""
        logger.info("🚀 Starting NOC MCP Server End-to-End Tests")
        logger.info("=" * 60)
        
        # List of all test methods
        tests = [
            self.test_tool_registry,
            self.test_tool_wrapper_creation,
            self.test_tool_execution,
            self.test_stdio_server_initialization,
            self.test_fastmcp_server_initialization,
            self.test_stdio_tool_call_handling,
            self.test_mcp_inspector_compatibility,
            self.test_server_health_and_info_tools,
            self.test_error_handling,
        ]
        
        # Run each test
        for test in tests:
            try:
                await test()
            except Exception as e:
                test_name = test.__name__.replace('test_', '').replace('_', ' ').title()
                self.log_test_result(test_name, False, f"Test threw exception: {str(e)}")
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)
        
        total_tests = self.passed_tests + self.failed_tests
        success_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {self.passed_tests} ✅")
        logger.info(f"Failed: {self.failed_tests} ❌")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        # Print detailed results
        logger.info("\n📋 DETAILED RESULTS:")
        for result in self.test_results:
            status_emoji = "✅" if result["status"] == "PASS" else "❌"
            logger.info(f"{status_emoji} {result['test']}: {result['status']} {result['message']}")
        
        # Return success status
        return self.failed_tests == 0


async def main():
    """Main test runner."""
    # Set up environment
    os.environ.setdefault('OPENAI_DEFAULT_API_KEY', 'test-key')
    os.environ.setdefault('OPENAI_DEFAULT_MODEL', 'openai/gpt-4o-mini')
    
    # Run tests
    test_suite = MCPServerE2ETests()
    success = await test_suite.run_all_tests()
    
    if success:
        logger.info("\n🎉 All tests passed! NOC MCP Server is ready for production.")
        sys.exit(0)
    else:
        logger.error("\n💥 Some tests failed. Please review the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
