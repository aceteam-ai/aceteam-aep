"""Tests for the aceteam-aep wrap CLI command."""

import subprocess
import sys


class TestWrapCLI:
    """Test the wrap subcommand."""

    def test_wrap_runs_command_and_exits(self):
        """wrap should run a command and return its exit code."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "aceteam_aep.proxy.cli",
                "wrap",
                "--no-safety",
                "--",
                "echo",
                "hello",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "hello" in result.stdout
        assert result.returncode == 0

    def test_wrap_shows_summary(self):
        """wrap should print AEP Summary after command exits."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "aceteam_aep.proxy.cli",
                "wrap",
                "--no-safety",
                "--",
                "echo",
                "done",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "AEP" in result.stdout

    def test_wrap_sets_env_vars(self):
        """wrap should set OPENAI_BASE_URL for the child process."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "aceteam_aep.proxy.cli",
                "wrap",
                "--no-safety",
                "--",
                sys.executable,
                "-c",
                "import os; print(os.environ.get('OPENAI_BASE_URL', 'NOT SET'))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert "http://localhost:" in result.stdout
        assert "/v1" in result.stdout

    def test_wrap_propagates_exit_code(self):
        """wrap should forward the child process exit code."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "aceteam_aep.proxy.cli",
                "wrap",
                "--no-safety",
                "--",
                sys.executable,
                "-c",
                "import sys; sys.exit(42)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 42

    def test_proxy_subcommand_help(self):
        """proxy help should work."""
        result = subprocess.run(
            [sys.executable, "-m", "aceteam_aep.proxy.cli", "proxy", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "proxy" in result.stdout.lower()

    def test_wrap_subcommand_help(self):
        """wrap help should work."""
        result = subprocess.run(
            [sys.executable, "-m", "aceteam_aep.proxy.cli", "wrap", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "wrap" in result.stdout.lower()
