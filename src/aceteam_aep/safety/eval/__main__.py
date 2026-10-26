"""Offline fixture commands; verifier model calls require an explicit client adapter."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import cast

from .dataset import (
    VerifierClient,
    assign_split,
    load_dataset,
    load_policies,
    verify_rows,
    write_rows,
)


def _split_rows(path: Path, *, write: bool) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    changed = False
    for row in rows:
        expected = assign_split(row["id"], row["category"], row["source"])
        if row["split"] != expected:
            changed = True
            if write:
                row["split"] = expected
    if changed and not write:
        raise SystemExit("committed split column differs from deterministic assignment")
    if write:
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _client_adapter(dotted: str) -> object:
    module_name, separator, name = dotted.partition(":")
    if not separator or not module_name or not name:
        raise ValueError("client adapter must be module:attribute")
    module = importlib.import_module(module_name)
    client = getattr(module, name)
    return client() if isinstance(client, type) else client


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    splits = commands.add_parser("splits", help="Recompute or check deterministic fixture splits")
    splits.add_argument("--labeled", type=Path, required=True)
    split_action = splits.add_mutually_exclusive_group(required=True)
    split_action.add_argument("--check", action="store_true")
    split_action.add_argument("--write", action="store_true")

    verify = commands.add_parser("verify", help="Run two operator-supplied verifier adapters")
    verify.add_argument("--catalog", type=Path, required=True)
    verify.add_argument("--labeled", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--client", required=True, help="module:attribute implementing verify")
    for field in ("model-a", "family-a", "model-b", "family-b", "generator-family"):
        verify.add_argument(f"--{field}", required=True)

    args = parser.parse_args(argv)
    if args.command == "splits":
        _split_rows(args.labeled, write=args.write)
        return
    if args.output.resolve() == args.labeled.resolve():
        parser.error("verify output must differ from the labeled input")
    policies = load_policies(args.catalog)
    rows = load_dataset(args.labeled, policies)
    verified = verify_rows(
        rows,
        policies,
        cast(VerifierClient, _client_adapter(args.client)),
        model_a=args.model_a,
        family_a=args.family_a,
        model_b=args.model_b,
        family_b=args.family_b,
        generator_family=args.generator_family,
    )
    write_rows(args.output, verified)


if __name__ == "__main__":
    main()
