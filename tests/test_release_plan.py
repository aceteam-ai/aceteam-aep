"""Offline checks for the CI release planner and version-bump command."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "release_plan.py"
SPEC = importlib.util.spec_from_file_location("release_plan", SCRIPT)
assert SPEC and SPEC.loader
release_plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_plan)


@pytest.fixture
def release_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "aceteam-aep"\nversion = "0.12.0"\n'
    )
    (tmp_path / "uv.lock").write_text('[[package]]\nname = "aceteam-aep"\nversion = "0.12.0"\n')
    (tmp_path / "CHANGELOG.md").write_text(
        "## [Unreleased]\n\n## [0.12.1rc1] - 2026-11-19\n\nRelease note.\n"
        "\n## [0.12.0] - 2026-10-26\n\nPrevious release.\n"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_sites_and_changelog_fail_closed(release_tree: Path) -> None:
    assert release_plan.version_sites() == "0.12.0"
    (release_tree / "uv.lock").write_text('[[package]]\nname = "aceteam-aep"\nversion = "0.11.9"\n')
    with pytest.raises(SystemExit, match="Version sites disagree"):
        release_plan.version_sites()
    with pytest.raises(SystemExit, match="lacks"):
        release_plan.changelog_section("0.12.1rc2")


def test_main_push_without_version_bump_is_noop(
    release_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        release_plan.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, (release_tree / "pyproject.toml").read_text(), ""
        ),
    )
    assert release_plan.plan("push", "a" * 40, "b" * 40, "aceteam-ai/aceteam-aep", "test") == {
        "run": "false",
        "published": "false",
        "version": "",
    }


def test_existing_release_does_not_republish(
    release_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(release_plan, "checked_tag", lambda version, sha: True)
    monkeypatch.setattr(release_plan, "pypi_metadata", lambda version: {"urls": []})
    monkeypatch.setattr(release_plan, "request_json", lambda url, token: {"id": 1})
    assert release_plan.plan(
        "workflow_dispatch", "", "b" * 40, "aceteam-ai/aceteam-aep", "test"
    ) == {"run": "false", "published": "true", "version": "0.12.0"}


def test_published_version_recovers_metadata_without_republishing(
    release_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(release_plan, "checked_tag", lambda version, sha: False)
    monkeypatch.setattr(release_plan, "pypi_metadata", lambda version: {"urls": []})
    monkeypatch.setattr(release_plan, "request_json", lambda url, token: None)
    assert release_plan.plan(
        "workflow_dispatch", "", "b" * 40, "aceteam-ai/aceteam-aep", "test"
    ) == {"run": "true", "published": "true", "version": "0.12.0"}


def test_conflicting_tag_fails_closed(release_tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def conflicting_tag(version: str, sha: str) -> bool:
        raise SystemExit("Existing tag points to another commit")

    monkeypatch.setattr(release_plan, "checked_tag", conflicting_tag)
    with pytest.raises(SystemExit, match="another commit"):
        release_plan.plan("workflow_dispatch", "", "b" * 40, "aceteam-ai/aceteam-aep", "test")


def test_bump_requires_changelog_and_changes_only_project_version(release_tree: Path) -> None:
    with pytest.raises(SystemExit, match="lacks"):
        release_plan.bump("0.12.1rc2")
    assert release_plan.bump("--rc") == "0.12.1rc1"
    assert 'version = "0.12.1rc1"' in (release_tree / "pyproject.toml").read_text()
    assert 'version = "0.12.0"' in (release_tree / "uv.lock").read_text()
    assert release_plan.version_order("0.12.1") > release_plan.version_order("0.12.1rc1")


def test_bump_script_cli_accepts_rc_flag(
    release_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    scripts = release_tree / "scripts"
    scripts.mkdir()
    shutil.copy2(SCRIPT, scripts / SCRIPT.name)
    shutil.copy2(SCRIPT.parent / "bump-version.sh", scripts / "bump-version.sh")
    bin_dir = release_tree / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env python3\n"
        "from pathlib import Path\n"
        "p=Path('uv.lock')\n"
        "p.write_text(p.read_text().replace('0.12.0', '0.12.1rc1'))\n"
    )
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    result = subprocess.run(
        ["bash", str(scripts / "bump-version.sh"), "--rc"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert release_plan.version_sites() == "0.12.1rc1"
