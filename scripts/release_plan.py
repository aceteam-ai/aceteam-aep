"""Fail-closed checks for the version-bump PyPI workflow.

This script never publishes, tags, or pushes. The GitHub workflow gives the
publishing job OIDC credentials only after these checks and the test jobs pass.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tomllib
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import NoReturn

PROJECT = "aceteam-aep"
VERSION_PATTERN = re.compile(r"^([0-9]+)\.([0-9]+)\.([0-9]+)(?:rc([1-9][0-9]*))?$")


def fail(message: str) -> NoReturn:
    raise SystemExit(message)


def parse_version(version: str) -> tuple[int, int, int, int]:
    match = VERSION_PATTERN.fullmatch(version)
    if not match:
        fail(f"Invalid release version: {version}")
    major, minor, patch, rc = match.groups()
    return int(major), int(minor), int(patch), int(rc) if rc else -1


def version_order(version: str) -> tuple[int, int, int, int, int]:
    major, minor, patch, rc = parse_version(version)
    return major, minor, patch, 0 if rc >= 0 else 1, rc


def project_version(text: str) -> str:
    value = tomllib.loads(text)["project"]["version"]
    parse_version(value)
    return value


def version_sites() -> str:
    version = project_version(Path("pyproject.toml").read_text())
    lock = tomllib.loads(Path("uv.lock").read_text())
    packages = [p for p in lock["package"] if p["name"] == PROJECT]
    if len(packages) != 1:
        fail("Missing or duplicate lockfile version site")
    if packages[0]["version"] != version:
        fail(f"Version sites disagree with pyproject.toml ({version})")
    return version


def changelog_section(version: str) -> str:
    text = Path("CHANGELOG.md").read_text()
    pattern = re.compile(r"^## \[" + re.escape(version) + r"\](?:[^\n]*)$", re.M)
    match = pattern.search(text)
    if not match:
        fail(f"CHANGELOG.md lacks ## [{version}]")
    next_section = re.search(r"^## \[", text[match.end() :], re.M)
    end = match.end() + next_section.start() if next_section else len(text)
    section = text[match.end() : end].strip()
    if not section:
        fail(f"CHANGELOG.md section for {version} is empty")
    return section + "\n"


def request_json(url: str, token: str | None = None) -> dict | None:
    headers = {"Accept": "application/json", "User-Agent": "aep-release"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        fail(f"Release metadata lookup failed: HTTP {error.code}")
    except (OSError, ValueError) as error:
        fail(f"Release metadata lookup failed: {type(error).__name__}")


def pypi_metadata(version: str) -> dict | None:
    return request_json(f"https://pypi.org/pypi/{PROJECT}/{version}/json")


def checked_tag(version: str, sha: str) -> bool:
    tag = f"v{version}"
    result = subprocess.run(
        ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return False
    if result.stdout.strip() != sha:
        fail(f"Existing {tag} tag points to another commit")
    return True


def plan(event: str, before: str, sha: str, repository: str, token: str) -> dict[str, str]:
    if event == "push":
        if not re.fullmatch(r"[0-9a-f]{40}", before) or before == "0" * 40:
            fail("Missing previous main commit")
        previous = subprocess.run(
            ["git", "show", f"{before}:pyproject.toml"],
            capture_output=True,
            text=True,
            check=True,
        )
        if project_version(previous.stdout) == project_version(Path("pyproject.toml").read_text()):
            return {"run": "false", "published": "false", "version": ""}
    elif event != "workflow_dispatch":
        fail(f"Unsupported event: {event}")

    version = version_sites()
    changelog_section(version)
    tagged = checked_tag(version, sha)
    published = pypi_metadata(version) is not None
    released = (
        request_json(f"https://api.github.com/repos/{repository}/releases/tags/v{version}", token)
        is not None
    )
    if released and not tagged:
        fail("GitHub release exists without a tag on this commit")
    return {
        "run": "true",
        "published": str(published).lower(),
        "version": version,
    }


def verify_dist(version: str, directory: Path, published: bool) -> None:
    files = sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith("."))
    if (
        len(files) != 2
        or sum(p.suffix == ".whl" for p in files) != 1
        or sum(p.name.endswith(".tar.gz") for p in files) != 1
    ):
        fail("Expected exactly one wheel and one sdist")
    expected_prefix = f"aceteam_aep-{version}"
    if any(not p.name.startswith(expected_prefix) for p in files):
        fail("Built artifact version does not match release version")
    for path in files:
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
        else:
            with tarfile.open(path, "r:gz") as archive:
                names = archive.getnames()
        if not any(name.endswith("aceteam_aep/__init__.py") for name in names):
            fail(f"Package code missing from {path.name}")
        if not any(name.endswith("METADATA") or name.endswith("PKG-INFO") for name in names):
            fail(f"Package metadata missing from {path.name}")
    if published:
        metadata = pypi_metadata(version)
        if metadata is None:
            fail("PyPI version disappeared during release")
        remote = {item["filename"]: item for item in metadata["urls"]}
        local = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
        if set(local) != set(remote):
            fail("Published artifacts do not match this commit's build")
        wheel = next(path for path in files if path.suffix == ".whl")
        if local[wheel.name] != remote[wheel.name]["digests"]["sha256"]:
            fail("Published wheel does not match this commit's build")
        sdist = next(path for path in files if path.name.endswith(".tar.gz"))
        published_sdist = remote[sdist.name]
        if local[sdist.name] != published_sdist["digests"]["sha256"]:
            # Hatchling tar headers can vary between identical builds. Verify
            # the published digest, then compare every unpacked file.
            url = published_sdist["url"]
            parsed = urllib.parse.urlparse(url)
            if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org":
                fail("Unexpected PyPI artifact host")
            with urllib.request.urlopen(url, timeout=20) as response:
                archive_bytes = response.read(25_000_001)
                final_host = urllib.parse.urlparse(response.url).hostname
            if final_host != "files.pythonhosted.org":
                fail("Unexpected PyPI artifact redirect")
            if (
                len(archive_bytes) > 25_000_000
                or hashlib.sha256(archive_bytes).hexdigest() != published_sdist["digests"]["sha256"]
            ):
                fail("Published sdist digest is invalid")
            with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
                remote_files = archive_contents(archive)
            with tarfile.open(sdist, "r:gz") as archive:
                local_files = archive_contents(archive)
            if remote_files != local_files:
                fail("Published sdist does not match this commit's build")


def archive_contents(archive: tarfile.TarFile) -> dict[str, bytes]:
    contents = {}
    for member in archive.getmembers():
        if not member.isfile():
            continue
        file = archive.extractfile(member)
        if file is None:
            fail("Cannot read sdist member")
        contents[member.name] = file.read()
    return contents


def bump(argument: str) -> str:
    current = version_sites()
    major, minor, patch, rc = parse_version(current)
    if argument == "--rc":
        target = (
            f"{major}.{minor}.{patch}rc{rc + 1}" if rc >= 0 else f"{major}.{minor}.{patch + 1}rc1"
        )
    elif argument == "--patch":
        target = f"{major}.{minor}.{patch + 1}"
    elif argument == "--minor":
        target = f"{major}.{minor + 1}.0"
    elif argument == "--major":
        target = f"{major + 1}.0.0"
    else:
        target = argument
    if version_order(target) <= version_order(current):
        fail("New version must be greater than current version")
    changelog_section(target)
    edits = [
        (Path("pyproject.toml"), r'(?m)^version = "[^"]+"$', f'version = "{target}"'),
    ]
    for path, pattern, replacement in edits:
        text = path.read_text()
        updated, count = re.subn(pattern, replacement, text, count=1)
        if count != 1:
            fail(f"Cannot update {path}")
        path.write_text(updated)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--event", required=True)
    plan_parser.add_argument("--before", default="")
    plan_parser.add_argument("--sha", required=True)
    plan_parser.add_argument("--repository", required=True)
    subparsers.add_parser("check")
    notes_parser = subparsers.add_parser("notes")
    notes_parser.add_argument("version")
    dist_parser = subparsers.add_parser("verify-dist")
    dist_parser.add_argument("version")
    dist_parser.add_argument("directory", type=Path)
    dist_parser.add_argument("--published", action="store_true")
    bump_parser = subparsers.add_parser("bump")
    bump_parser.add_argument("version_or_flag")
    args = parser.parse_args()
    if args.command == "plan":
        values = plan(args.event, args.before, args.sha, args.repository, os.environ["GH_TOKEN"])
        with open(os.environ["GITHUB_OUTPUT"], "a") as output:
            for key, value in values.items():
                output.write(f"{key}={value}\n")
    elif args.command == "check":
        print(version_sites())
    elif args.command == "notes":
        sys.stdout.write(changelog_section(args.version))
    elif args.command == "verify-dist":
        verify_dist(args.version, args.directory, args.published)
    else:
        print(bump(args.version_or_flag))


if __name__ == "__main__":
    main()
