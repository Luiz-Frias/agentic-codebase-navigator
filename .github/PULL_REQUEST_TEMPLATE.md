## Description

<!-- Briefly describe what this PR does -->

## Type of Change

<!-- Select the type(s) of change -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update
- [ ] Refactoring
- [ ] Test improvement
- [ ] Dependency update

## Related Issue(s)

<!-- Link to related issues: Closes #123, Relates to #456 -->

Closes #

## Testing

<!-- Describe how you tested this change -->

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] E2E tests added/updated
- [ ] Manual testing performed

**Test Results:**

```text
[Paste test output here]
```

## Checklist

### Code Quality

- [ ] Code follows the project's style guidelines (`ruff format` + `ruff check --fix`)
- [ ] No new type errors (`ty check` for domain layer, `basedpyright` for all)
- [ ] All tests pass (`pytest`)
- [ ] Test coverage maintained/improved (≥90% requirement)
- [ ] No security vulnerabilities introduced (`bandit -r src/rlm`)

### Architecture

- [ ] Changes respect hexagonal architecture (domain → application → api → adapters)
- [ ] Domain layer changes have zero external dependencies
- [ ] New ports/adapters follow established patterns
- [ ] No circular dependencies introduced
- [ ] Dependency inversion maintained

### Documentation

- [ ] Commit messages follow [Conventional Commits](docs/contributing/commit-conventions.md)
- [ ] Code changes are documented (docstrings, type hints)
- [ ] User-facing changes documented in docstrings/comments
- [ ] README or docs updated if needed
- [ ] No secrets or credentials committed

### Pre-Commit & CI

- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] No new `gitleaks` or `detect-secrets` warnings
- [ ] Ready for CI pipeline
- [ ] All GitHub Actions passing

## Breaking Changes

<!-- Describe any breaking changes and migration path -->

- [ ] No breaking changes
- [ ] Breaking changes (see details below)

**If breaking:**

- What changed and why?
- How should users migrate?

## Screenshots / Examples (if applicable)

<!-- For UI changes, feature additions, or documentation updates -->

## Additional Notes

<!-- Any other context or considerations -->
