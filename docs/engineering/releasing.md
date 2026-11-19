# Releasing aceteam-aep

A reviewed version-bump PR is the release gate. The merged change to
`pyproject.toml` starts `.github/workflows/publish.yml` on `main`. This workflow
checks the version against `uv.lock` and `CHANGELOG.md`, runs tests, builds the
wheel and sdist, publishes to PyPI through OIDC with attestations, creates an
annotated tag and GitHub release, and then builds the aep-proxy image from the
same commit. A main push without a version change does nothing.

Before the first CI release, Jason must configure the PyPI trusted publisher for
repository `aceteam-ai/aceteam-aep`, workflow `publish.yml`, and environment
`pypi`. Keep these names unchanged. The workflow has no token secret and the
`pypi` environment has no extra approval gate. The existing `PYPI_API_TOKEN`
secret is retained only for an authorized emergency recovery, outside CI.

To prepare a release:

1. Add a nonempty `## [VERSION]` section to `CHANGELOG.md`.
2. Run `scripts/bump-version.sh VERSION` or a bump flag (`--rc`, `--patch`,
   `--minor`, `--major`). It changes `pyproject.toml` and refreshes `uv.lock`;
   it never commits, tags, pushes, or publishes.
3. Review those files in a PR. After the PR merges, check the publish workflow,
   PyPI attestations, the annotated tag, GitHub release, and image tag.

For a recovery rerun, use the workflow's manual dispatch on `main`. An already
published version is never uploaded again. The recovery path verifies that the
built wheel and sdist match PyPI, verifies or creates the tag and release, and
reruns the image build. A tag pointing to a different commit fails closed.

The old `scripts/release.sh` is removed because it could publish and push from a
laptop. Retain `PYPI_API_TOKEN` only for the authorized emergency procedure.
