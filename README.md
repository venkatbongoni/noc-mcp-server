# NOC Internal MCP Server for All Internal Tools

A comprehensive Model Context Protocol (MCP) server implementation with DSPY framework integration for Network Operations Center (NOC) tools including syslog analysis, Splunk integration, command execution, memory management, and knowledge base operations.

## 🎯 Overview

This project builds an MCP server that exposes network operations tools through a standardized protocol, enabling seamless integration with DSPY agents and other MCP-compatible clients. The server provides both simplified demonstration capabilities and full production-ready functionality.

## 🏗️ Architecture

### Core Components

```
noc-mcp-server/
├── core/                    # Core MCP server functionality
│   ├── mcp_server.py       # Main MCP server implementation
│   ├── mcp_client.py       # MCP client for connecting to servers
│   ├── mcp_agent.py        # DSPY agent with MCP integration
│   ├── mcp_tools.py        # Full MCP tool wrappers
│   ├── mcp_tools_simple.py # Simplified tool wrappers for testing
│   └── tool_registry.py    # Tool package management
├── agents/                  # Agent implementations
│   └── syslog_tools.py     # Original syslog tool definitions
├── auth/                    # Authentication management
│   └── auth_manager.py     # Authentication handling
├── config/                  # Configuration files
│   └── tool_packages.yaml  # Tool package definitions
├── tools/                   # Original tool implementations (DO NOT MODIFY)
├── lib/                     # Shared libraries
│   ├── datastore.py        # Memory management
│   └── datastore_interface.py
├── tests/                   # Test suite
└── scripts/                 # Utility scripts
```

### Tool Packages

The server supports different tool packages defined in `config/tool_packages.yaml`:

- **syslog**: System log analysis tools
  - `READ_NEW_LOG_MESSAGES`: Read recent syslog data
  - `HIGHEST_SERVICE_IMPACT`: Evaluate operational impact
  - `QUERY_DEVICES`: Execute commands on network devices
  - `HOLISTIC_CAUSAL_REASONING`: Root cause analysis
  - `EVALUATE_DRAFT_ANSWER`: Analysis completeness evaluation
  - `SHORT_TERM_MEMORY`: Memory operations

- **rag**: Retrieval-Augmented Generation tools
  - `LOOK_UP_FROM_ICS_SQL_SYSTEM`: ICS SQL system lookup
  - `LOOK_UP_FROM_ICS_RAGSYSTEM`: ICS RAG system lookup

## 🚀 Quick Start

### Prerequisites

- Python >3.13, <3.14
- Virtual environment (recommended)

### Installation

1. **Clone and setup the environment:**
   ```bash
   cd noc-mcp-server
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Run basic functionality test:**
   ```bash
   python3 test_basic.py
   ```

### Running the Server

#### Option 1: Simplified Server (Recommended for Testing)

Run the simplified server with automatic demo:
```bash
python3 simple_server.py --auto-demo
```

Run interactive simplified server:
```bash
python3 simple_server.py
```

Available commands in interactive mode:
- `workflow <query>` - Run full analysis workflow
- `call <tool> <key=value>` - Call specific tool
- `memory list` - List memory contents
- `memory get <key>` - Get memory value
- `help` - Show help
- `quit` - Exit

#### Option 2: Full MCP Server (Requires Dependencies)

```bash
python -m core.mcp_server --port 8075 --tool-package syslog
```

### Example Usage

#### Simplified Server Demo

```bash
$ python3 simple_server.py --auto-demo

============================================================
AUTO DEMO: High CPU utilization detected
============================================================

READ_LOGS:
  Retrieved 3 log messages for query: High CPU utilization detected

IMPACT_ANALYSIS:
  Impact analysis for 'Impact analysis for: High CPU utilization detected': Found 1 high-impact events out of 3 total events

DEVICE_QUERY:
  Device query for 'Device diagnostics for: High CPU utilization detected': Checked 3 devices
  router1: up
  switch1: down
  router2: up

ROOT_CAUSE:
  Root cause analysis for 'Root cause for: High CPU utilization detected':
  • High impact events detected: 1
  • Device failures detected: switch1
  • Link state changes detected: 1 events

Recommendation: Investigate device failures and link state changes

EVALUATION:
  Draft answer evaluation for 'Evaluate completeness for: High CPU utilization detected':
Completeness Score: 100/100
...
```

#### Interactive Demo

```bash
$ python3 simple_server.py

noc> workflow Network latency issues
Running workflow: Network latency issues
...

noc> call SHORT_TERM_MEMORY action=list
Memory: Memory keys: context, high_impact, device_data

noc> memory get context
Memory: ['2024-01-15 10:00:01 router1: %BGP-5-ADJCHANGE...', ...]
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file (automatically created by setup script):

```bash
# Server settings
NOC_MCP_PORT=8075
NOC_TOOL_PACKAGE=syslog

# Logging
LOG_LEVEL=INFO

# Syslog configuration
SYSLOG_API_BASE=http://localhost:8080
SYSLOG_HOST=localhost
SYSLOG_BUFFER=main

# OpenAI Configuration (optional)
OPENAI_API_KEY=your_api_key_here
MODEL_NAME=gpt-4o-mini
```

### Tool Package Configuration

Edit `config/tool_packages.yaml` to define custom tool packages:

```yaml
syslog:
  - READ_NEW_LOG_MESSAGES
  - HIGHEST_SERVICE_IMPACT
  - QUERY_DEVICES
  - HOLISTIC_CAUSAL_REASONING
  - EVALUATE_DRAFT_ANSWER
  - SHORT_TERM_MEMORY

custom_package:
  - CUSTOM_TOOL_1
  - CUSTOM_TOOL_2
```

## 🧪 Testing

### Basic Functionality Test

```bash
python3 test_basic.py
```

### Unit Tests

```bash
python -m pytest tests/ -v
```

### Integration Tests

```bash
python integrated_demo.py --duration 30
```

## 🏭 Production Deployment

### Full Dependencies Installation

For production deployment with all features:

```bash
pip install -r requirements.txt
pip install fastmcp mcp dspy-ai
```

### Running with Full MCP Support

```bash
python -m core.mcp_server --port 8075 --tool-package syslog --log-level INFO
```

## 📊 Monitoring and Observability

### Logging

The server provides comprehensive logging:
- Request/response logging
- Tool execution tracking
- Error monitoring
- Performance metrics

### Health Checks

```bash
# Via simplified server
noc> call health_check

# Via MCP client
curl http://localhost:8075/health
```

## 🔌 Integration Examples

### DSPY Agent Integration

```python
from core.mcp_agent import NOCMCPAgent

async def analyze_network_issue():
    async with NOCMCPAgent("http://localhost:8075") as agent:
        result = await agent.analyze_network_issue("BGP neighbor down")
        print(result["final_analysis"])
```

### MCP Client Integration

```python
from core.mcp_client import NOCMCPClient

async def use_tools():
    async with NOCMCPClient("http://localhost:8075") as client:
        tools = client.get_available_tools()
        result = await client.call_tool("READ_NEW_LOG_MESSAGES", 
                                      query="Recent network errors")
```

## 🛠️ Development

### Adding New Tools

1. **Create tool wrapper in `core/mcp_tools_simple.py`:**

```python
class NewToolWrapper(MCPToolWrapper):
    def __init__(self):
        super().__init__("NEW_TOOL", "Description of new tool")
    
    async def execute(self, **kwargs) -> str:
        # Tool implementation
        return "Tool result"
```

2. **Register in factory method:**

```python
tool_map = {
    "NEW_TOOL": NewToolWrapper,
    # ... existing tools
}
```

3. **Add to tool package in `config/tool_packages.yaml`:**

```yaml
syslog:
  - NEW_TOOL
  # ... existing tools
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure virtual environment is activated and dependencies installed
2. **Port Conflicts**: Change port with `--port` argument
3. **Tool Not Found**: Check tool name in `config/tool_packages.yaml`
4. **Memory Issues**: Use `memory clear` or restart server

### Debug Mode

```bash
python3 simple_server.py --log-level DEBUG
```

## 📚 API Reference

### Tool Signatures

All tools follow the MCP tool signature pattern:

```python
async def tool_name(**kwargs) -> str:
    """Tool description.
    
    Args:
        query: Analysis query (optional)
        **kwargs: Tool-specific parameters
        
    Returns:
        str: Tool execution result
    """
```

### Memory Operations

```python
# Set memory
await call_tool("SHORT_TERM_MEMORY", action="set", key="data", value="value")

# Get memory  
await call_tool("SHORT_TERM_MEMORY", action="get", key="data")

# List keys
await call_tool("SHORT_TERM_MEMORY", action="list")

# Clear all
await call_tool("SHORT_TERM_MEMORY", action="clear")
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Make changes and test: `python3 test_basic.py`
4. Commit changes: `git commit -am 'Add new feature'`
5. Push to branch: `git push origin feature/new-feature`
6. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check troubleshooting section
2. Run basic tests: `python3 test_basic.py`
3. Check logs for error details
4. Create GitHub issue with detailed description

---

**Status**: ✅ Basic functionality implemented and tested
**Next Steps**: Full MCP server implementation with external dependencies
