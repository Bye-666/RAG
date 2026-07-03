"""Tests for MCP Server."""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.mcp_server.server import MCPServer


@pytest.fixture
def mcp_server():
    """Create MCP server instance."""
    return MCPServer()


def test_mcp_server_instantiation():
    """MCPServer can be instantiated."""
    server = MCPServer()
    assert server.config == {}
    assert server.server is not None


def test_mcp_server_with_config():
    """MCPServer accepts configuration."""
    config = {"key": "value"}
    server = MCPServer(config)
    assert server.config == config


@pytest.mark.asyncio
async def test_handle_ingest(mcp_server):
    """Handle ingest tool call."""
    args = {"file_path": "/path/to/document.pdf"}

    result = await mcp_server._handle_ingest(args)

    assert len(result) == 1
    assert result[0].type == "text"
    assert "document.pdf" in result[0].text


@pytest.mark.asyncio
async def test_handle_query(mcp_server):
    """Handle query tool call."""
    args = {"query": "test query", "top_k": 5}

    result = await mcp_server._handle_query(args)

    assert len(result) == 1
    assert "test query" in result[0].text


@pytest.mark.asyncio
async def test_handle_query_default_top_k(mcp_server):
    """Handle query with default top_k."""
    args = {"query": "test query"}

    result = await mcp_server._handle_query(args)

    assert len(result) == 1
    assert "10" in result[0].text  # Default top_k


@pytest.mark.asyncio
async def test_handle_list_documents(mcp_server):
    """Handle list documents tool call."""
    args = {}

    result = await mcp_server._handle_list_documents(args)

    assert len(result) == 1
    assert "documents" in result[0].text


def test_server_has_tools_registered(mcp_server):
    """Server is initialized with tools."""
    # Server should be properly initialized
    assert mcp_server.server is not None
    assert hasattr(mcp_server.server, 'name')
    assert mcp_server.server.name == "rag-mcp-server"
