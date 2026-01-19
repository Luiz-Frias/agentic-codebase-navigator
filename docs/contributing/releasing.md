# Release Process

This document describes the release workflow for `agentic-codebase-navigator`.

## Overview

We follow a **dev → staging → main** branching model with semantic versioning:

```
feature branches → dev → staging → main
                         ↓
                     RC tags (v1.0.0-rc.1)
                                    ↓
                               Final tags (v1.0.0)
                                    ↓
                                 PyPI
```

## Prerequisites

### PyPI Setup (One-Time)

1. **Create PyPI account**: [pypi.org/account/register](https://pypi.org/account/register/)
2. **Create API token**: [pypi.org/manage/account/token](https://pypi.org/manage/account/token/)
3. **Add GitHub secret**: Repository → Settings → Secrets → `PYPI_API_TOKEN`

For TestPyPI (recommended for first-time setup):

1. **Create TestPyPI account**: [test.pypi.org/account/register](https://test.pypi.org/account/register/)
2. **Create API token**: [test.pypi.org/manage/account/token](https://test.pypi.org/manage/account/token/)
3. **Add GitHub secret**: `TEST_PYPI_API_TOKEN`

### GitHub Environments (Optional but Recommended)

Create environments for deployment protection:

1. Go to Repository → Settings → Environments
2. Create `testpypi` environment (for RC releases)
3. Create `pypi` environment (for final releases)
4. Add protection rules (e.g., required reviewers, wait timers)

## Release Workflow

### Step 1: Prepare Release on `dev`

```bash
# Ensure you're on dev and up-to-date
git checkout dev
git pull origin dev

# Run full quality gates locally
uv run --group dev ruff format --check .
uv run --group dev ruff check .
uv run --group test pytest -m unit
uv run --group test pytest -m integration
uv run --group test pytest -m e2e
uv run --group test pytest -m packaging
```

### Step 2: Update Version and Changelog

Edit `pyproject.toml`:

```toml
[project]
version = "X.Y.Z"  # e.g., "1.0.0"
```

Edit `CHANGELOG.md`:

```markdown
## X.Y.Z

### Features
- New feature description

### Bug Fixes
- Bug fix description

### Breaking Changes
- Breaking change description (if any)
```

Commit the version bump:

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore(release): vX.Y.Z"
git push origin dev
```

### Step 3: Create Release Candidate (Optional)

For significant releases, test with a release candidate first:

```bash
# Tag RC on dev
git tag v1.0.0-rc.1
git push origin v1.0.0-rc.1
```

This triggers:
1. Quality gates (CI)
2. Build artifacts
3. Publish to TestPyPI

Verify the RC on TestPyPI:

```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    agentic-codebase-navigator==1.0.0rc1

# Test it works
python -c "from rlm import RLM, __version__; print(__version__)"
```

### Step 4: Merge to Staging

```bash
# Create PR: dev → staging
gh pr create --base staging --head dev --title "Release v1.0.0"

# After review/approval, merge
gh pr merge --squash
```

### Step 5: Final Merge to Main

```bash
# Pull staging and merge to main
git checkout main
git pull origin main
git merge staging --no-ff -m "Merge staging for v1.0.0"
git push origin main
```

### Step 6: Create Final Release Tag

```bash
# Tag the release on main
git tag v1.0.0
git push origin v1.0.0
```

This triggers the release workflow:
1. Quality gates run
2. Build wheel + sdist
3. Publish to PyPI
4. Create GitHub Release with artifacts

### Step 7: Verify Release

```bash
# Wait for workflow to complete (~5 min)
gh run watch

# Install from PyPI
pip install agentic-codebase-navigator==1.0.0

# Verify
python -c "from rlm import __version__; print(__version__)"
```

## Manual Release (Alternative)

If GitHub Actions isn't available, you can release manually:

```bash
# 1. Build artifacts
uv build --wheel --sdist

# 2. Verify build
ls -la dist/
# Should show:
# agentic_codebase_navigator-1.0.0-py3-none-any.whl
# agentic_codebase_navigator-1.0.0.tar.gz

# 3. Publish to TestPyPI first
uv publish --index-url https://test.pypi.org/simple/ dist/*
# Enter username: __token__
# Enter password: <your-test-pypi-token>

# 4. Verify on TestPyPI
pip install --index-url https://test.pypi.org/simple/ agentic-codebase-navigator

# 5. Publish to PyPI
uv publish dist/*
# Enter username: __token__
# Enter password: <your-pypi-token>
```

## Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0 → 2.0.0): Breaking API changes
- **MINOR** (1.0.0 → 1.1.0): New features, backward compatible
- **PATCH** (1.0.0 → 1.0.1): Bug fixes, backward compatible

Pre-release versions:

- **RC** (Release Candidate): `v1.0.0-rc.1`, `v1.0.0-rc.2`
- **Alpha/Beta**: `v1.0.0-alpha.1`, `v1.0.0-beta.1` (if needed)

## Troubleshooting

### "Package already exists" Error

PyPI doesn't allow re-uploading the same version. Options:
1. Bump version (e.g., 1.0.1)
2. Delete the package on PyPI (only if not yet downloaded)

### Build Fails

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Rebuild
uv build --wheel --sdist
```

### Tests Fail in CI

```bash
# Run the full test suite locally
uv run --group test pytest

# Check specific failing markers
uv run --group test pytest -m unit -v
```

### Token Issues

- Ensure token has upload permissions (create with "Upload" scope)
- Check token isn't expired
- Verify secret name matches workflow (`PYPI_API_TOKEN`)

## Release Checklist

Before tagging a release, verify:

- [ ] All tests pass locally
- [ ] CI is green on `dev`
- [ ] Version bumped in `pyproject.toml`
- [ ] `CHANGELOG.md` updated with release notes
- [ ] `README.md` reflects current features
- [ ] No uncommitted changes
- [ ] PR merged to `main` (via `staging`)

## Rollback

If a release has issues:

1. **Yank on PyPI** (prevents new installs):
   ```bash
   # Via web: pypi.org/manage/project/.../release/X.Y.Z/
   # Click "Options" → "Yank"
   ```

2. **Delete GitHub Release** (if not widely distributed)

3. **Fix forward**: Create a patch release with the fix

> **Note**: You cannot delete a version from PyPI; you can only yank it to hide from default searches.

## See Also

- [uv publish documentation](https://docs.astral.sh/uv/guides/publish/)
- [PyPI publishing guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)
