# NOC MCP Server - FastMCP Installation Guide

## 🚀 Quick Start (No Dependencies)

For immediate testing without external dependencies:

```bash
cd noc-mcp-server
python3 simple_server.py
```

This runs the simplified server with all tools working, just without full MCP protocol support.

## 📦 Full Installation with FastMCP

For complete MCP protocol support and production deployment:

### 1. Install FastMCP Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install FastMCP and MCP dependencies
pip install fastmcp>=0.2.0
pip install mcp>=1.0.0

# Or install all optional dependencies
pip install -e ".[mcp]"
```

### 2. Run Full MCP Server

```bash
# Start the production MCP server
python3 -m core.mcp_server --port 8075 --tool-package syslog

# Or with custom settings
python3 -m core.mcp_server --port 8080 --tool-package rag --log-level DEBUG
```

## ⚠️ Troubleshooting AsyncIO Issues

If you encounter `RuntimeError: Already running asyncio in this thread`, this has been fixed in the latest version. The server now properly handles FastMCP's event loop requirements.

### Fixed Issues:
- ✅ AsyncIO event loop conflicts resolved
- ✅ FastMCP compatibility improved  
- ✅ Proper sync/async handling implemented

### Verification Commands:

```bash
# Test import (should work without dependencies)
python3 -c "from core.mcp_server import NOCMCPServer; print('✅ Import successful')"

# Test simplified server (no FastMCP required)
echo "quit" | python3 simple_server.py

# Test full server (requires FastMCP)
python3 -m core.mcp_server --help
```

## 🎯 Deployment Options

### Option 1: Simplified Server (Recommended for Development)
- ✅ No external dependencies
- ✅ All NOC tools functional
- ✅ Interactive demo mode
- ✅ Workflow testing

```bash
python3 simple_server.py --port 8075
```

### Option 2: Full MCP Server (Production)
- ✅ Complete MCP protocol support
- ✅ Client/server communication
- ✅ Integration with MCP ecosystem
- ⚠️ Requires FastMCP installation

```bash
python3 -m core.mcp_server --port 8075 --tool-package syslog
```

## 📊 Current Status

Both deployment options are fully functional:

- **Simplified Server**: 100% ready, all tests passing
- **Full MCP Server**: Ready with correct dependencies
- **Tool Functionality**: All 6 NOC tools working in both modes
- **AsyncIO Issues**: ✅ Fixed and resolved

Choose the option that best fits your needs!
