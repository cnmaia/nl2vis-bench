"""Tests for CLI application."""
import sys
import pytest
import tempfile
import os
from pathlib import Path
from typer.testing import CliRunner

# Import the CLI app
from nl2vis_bench.cli import app as cli_app

runner = CliRunner()


@pytest.fixture
def temp_config_dir(monkeypatch, tmp_path):
    """Use a temporary directory for config."""
    # Get the actual module from sys.modules (not the Typer app object)
    cli_module = sys.modules['nl2vis_bench.cli.app']

    monkeypatch.setattr(cli_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli_module, "CONFIG_FILE", tmp_path / "config.json")
    monkeypatch.setattr(cli_module, "CATALOG_DB", tmp_path / "catalog.db")
    yield tmp_path


def test_version_command():
    result = runner.invoke(cli_app, ["version"])
    assert result.exit_code == 0
    assert "nl2vis-bench version" in result.output


def test_config_show_empty(temp_config_dir):
    result = runner.invoke(cli_app, ["config", "--show"])
    assert result.exit_code == 0
    assert "No configuration found" in result.output


def test_config_set_provider(temp_config_dir):
    result = runner.invoke(cli_app, ["config", "--provider", "gemini"])
    assert result.exit_code == 0
    assert "Configuration saved" in result.output


def test_config_set_gemini_key(temp_config_dir):
    result = runner.invoke(cli_app, ["config", "--gemini-key", "test-key-123"])
    assert result.exit_code == 0
    assert "Configuration saved" in result.output

    # Verify key was saved
    result = runner.invoke(cli_app, ["config", "--show"])
    assert "***" in result.output  # Key is masked


def test_config_invalid_provider(temp_config_dir):
    result = runner.invoke(cli_app, ["config", "--provider", "invalid"])
    assert result.exit_code == 1
    assert "Invalid provider" in result.output


def test_list_empty(temp_config_dir):
    result = runner.invoke(cli_app, ["list"])
    assert result.exit_code == 0
    assert "No datasets indexed yet" in result.output


def test_index_file_not_found(temp_config_dir):
    result = runner.invoke(cli_app, ["index", "/nonexistent/file.xlsx"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_index_and_list(temp_config_dir):
    # Create a test file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode='w') as f:
        f.write("timestamp,value\n2024-01-01,10\n2024-01-02,20\n")
        temp_file = f.name

    try:
        # Index the file
        result = runner.invoke(cli_app, ["index", temp_file, "--name", "test_data"])
        assert result.exit_code == 0, f"output: {result.output}"
        assert "Indexed dataset" in result.output

        # List should show it
        result = runner.invoke(cli_app, ["list"])
        assert result.exit_code == 0
        assert "test_data" in result.output
    finally:
        os.unlink(temp_file)


def test_index_unsupported_type(temp_config_dir):
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        temp_file = f.name

    try:
        result = runner.invoke(cli_app, ["index", temp_file])
        assert result.exit_code == 1
        assert "No handler" in result.output
    finally:
        os.unlink(temp_file)


def test_query_dataset_not_found(temp_config_dir):
    result = runner.invoke(cli_app, ["query", "nonexistent", "What is the average?"])
    assert result.exit_code == 1
    assert "not found" in result.output
