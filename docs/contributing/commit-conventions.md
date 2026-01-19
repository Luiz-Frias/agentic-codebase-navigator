# Commit Conventions

RLM follows [Conventional Commits](https://www.conventionalcommits.org/) for consistent, meaningful commit history.

## Format

```
type(scope): subject

- Bullet point explaining what and why
- Another bullet point
- Up to 6 bullets total
```

## Quick Reference

### Types

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat(api): add async completion support` |
| `fix` | Bug fix | `fix(broker): handle connection timeout` |
| `docs` | Documentation | `docs(readme): add installation guide` |
| `style` | Code style (no logic change) | `style(domain): fix line length` |
| `refactor` | Code refactoring | `refactor(orchestrator): extract helper method` |
| `perf` | Performance improvement | `perf(codec): optimize frame encoding` |
| `test` | Add/update tests | `test(unit): add orchestrator edge cases` |
| `build` | Build system changes | `build(deps): update openai to 2.14` |
| `ci` | CI/CD changes | `ci(workflow): add packaging job` |
| `chore` | Maintenance tasks | `chore(release): bump version to 1.1.0` |

### Scopes

| Scope | Description |
|-------|-------------|
| `core` | Core functionality |
| `domain` | Domain layer |
| `ports` | Port interfaces |
| `app` | Application layer |
| `api` | API layer |
| `adapters` | Adapter implementations |
| `infra` | Infrastructure layer |
| `cli` | Command-line interface |
| `deps` | Dependencies |
| `tooling` | Development tools |
| `docs` | Documentation |
| `tests` | Test suite |

## Rules

### Subject Line

- **Max 50 characters**
- **Imperative mood** ("add" not "added", "fix" not "fixed")
- **No period** at the end
- **Lowercase** first letter

```
✅ feat(api): add tool calling support
✅ fix(broker): handle empty response
❌ feat(api): Added tool calling support.
❌ fix(broker): Handles empty response
```

### Body

- **3-6 bullet points** explaining what and why
- **72 character wrap** per line
- Use `-` for bullets
- Focus on **why**, not just what

```
feat(domain): add stopping policy protocol

- Enable custom iteration termination logic
- Support EIG-based and entropy-based stopping
- Integrate with orchestrator via protocol pattern
- Add default implementation for max_iterations
```

## AI-Assisted Commits

Use `just commit-msg` to generate commit messages:

```bash
# Stage changes
git add .

# Generate message (copies to clipboard)
just commit-msg

# Review the generated message, then commit
git commit -m "$(pbpaste)"  # macOS
# or
just commit-ai  # Interactive commit
```

The AI generates messages following these conventions by analyzing:
- Git diff of staged changes
- Recent commit history for style consistency

## Examples

### Feature Addition

```
feat(adapters): add Gemini LLM provider

- Implement GeminiAdapter with tool calling support
- Add google-genai as optional dependency
- Support function declaration format conversion
- Register in DefaultLLMRegistry
```

### Bug Fix

```
fix(broker): prevent deadlock on batched timeout

- Add cancellation grace period for pending tasks
- Shield cleanup from task cancellation
- Log warning instead of raising during cleanup
- Fixes #123
```

### Refactoring

```
refactor(orchestrator): extract tool execution logic

- Move tool execution to dedicated method
- Improve error handling for tool failures
- Reduce cognitive complexity of main loop
- No behavior changes
```

### Documentation

```
docs(testing): add comprehensive testing guide

- Document all pytest markers and their purposes
- Add examples for writing unit/integration/e2e tests
- Include live LLM test opt-in instructions
- Add troubleshooting section
```

### Breaking Changes

```
feat(api)!: rename create_rlm to build_rlm

BREAKING CHANGE: `create_rlm()` is now `build_rlm()`

- Align naming with factory pattern conventions
- Old name deprecated, will be removed in 2.0
- Update all documentation and examples
```

### Multiple Changes

If a commit touches multiple areas, use the most significant type/scope:

```
feat(domain): add context compression protocol

- Define ContextCompressor protocol in agent_ports
- Add NoOpContextCompressor default implementation
- Integrate with orchestrator for nested results
- Add unit tests for compression logic
- Update extending.md documentation
```

## Commit Hygiene

### Atomic Commits

Each commit should be **one logical change**:

```
✅ Single commit: "feat(api): add async completion"
❌ Mixed commit: "feat(api): add async + fix typo + update docs"
```

### Rebase Before Merging

Keep history clean by rebasing feature branches:

```bash
git checkout feature/my-feature
git rebase dev
git push --force-with-lease
```

### Squash When Appropriate

For WIP commits, squash before merging:

```bash
# Interactive rebase to squash
git rebase -i HEAD~5

# In editor, change 'pick' to 'squash' for WIP commits
```

## Validation

### Pre-commit Hook

The `commitizen` hook validates commit messages:

```bash
# If commit is rejected
$ git commit -m "bad message"
commitizen check failed: message does not follow conventional commits

# Correct format
$ git commit -m "feat(api): add new feature"
✓ Commit message is valid
```

### Manual Validation

```bash
# Check a commit message
echo "feat(api): add feature" | cz check
```

## See Also

- [Development Setup](development-setup.md) — Local development workflow
- [Releasing](releasing.md) — Release process
- [Conventional Commits](https://www.conventionalcommits.org/) — Full specification
