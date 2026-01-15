# =============================================================================
# JUSTFILE - semantic-code-agents-rs
# =============================================================================
# Task runner for quality gates, development, and CI/CD.
# Run `just` or `just --list` to see available commands.
# Docs: https://github.com/casey/just
# =============================================================================

# Use bash for shell commands
set shell := ["bash", "-cu"]

# Default recipe - show help
default:
    @just --list --unsorted

# =============================================================================
# PRE-COMMIT HOOKS (full suite)
# =============================================================================

# Run all pre-commit hooks on staged files
pc:
    @echo "→ Running pre-commit hooks on staged files..."
    pre-commit run

# Run all pre-commit hooks on all files
pc-all:
    @echo "→ Running pre-commit hooks on all files..."
    pre-commit run --all-files

# Run pre-push hooks on all files
pc-push:
    @echo "→ Running pre-push hooks on all files..."
    pre-commit run --all-files --hook-stage pre-push

# Stage all changes and run all pre-commit hooks
pc-staged:
    @echo "→ Staging all changes and running pre-commit..."
    git add -A
    pre-commit run --all-files

# Full gate: stage, run pre-commit, then pre-push hooks
pc-full:
    @echo "→ Running full pre-commit + pre-push gate..."
    git add -A
    pre-commit run --all-files
    pre-commit run --all-files --hook-stage pre-push
    @echo "✓ All pre-commit and pre-push hooks passed"

# Install all pre-commit hook types
pc-install:
    @echo "→ Installing pre-commit hooks..."
    pre-commit install --install-hooks
    pre-commit install --hook-type commit-msg
    pre-commit install --hook-type pre-push
    @echo "✓ All hook types installed"

# Update pre-commit hooks to latest versions
pc-update:
    @echo "→ Updating pre-commit hooks..."
    pre-commit autoupdate
    @echo "✓ Hooks updated"

# =============================================================================
# SETUP
# =============================================================================

# Install development tools
setup:
    @echo "→ Installing development tools..."
    cargo install cargo-watch cargo-audit cargo-deny cargo-machete cargo-tarpaulin taplo-cli
    @echo "→ Setting up pre-commit..."
    pre-commit install --install-hooks
    pre-commit install --hook-type commit-msg
    pre-commit install --hook-type pre-push
    @echo "✓ Development environment ready"

# =============================================================================
# GIT / COMMITS
# =============================================================================

# Generate AI commit message following project conventions (requires staged changes)
commit-msg:
    #!/usr/bin/env bash
    set -euo pipefail

    # Check for staged changes
    if git diff --cached --quiet; then
        echo "❌ No staged changes. Stage files first with 'git add'"
        exit 1
    fi

    # Get staged diff
    DIFF=$(git diff --cached --stat && echo "" && git diff --cached)

    # Build the prompt with project conventions
    PROMPT=$(cat <<'PROMPT_EOF'
    Generate a git commit message following these EXACT conventions:

    FORMAT:
    <type>(<scope>): <subject>

    <body>

    STRUCTURE:
    1. SUBJECT LINE: <type>(<scope>): <subject>
       - Max 50 chars, imperative mood ("add" not "added"), no period
    2. BLANK LINE after subject
    3. BODY: 3-6 bullet points explaining WHAT changed and WHY
       - Start each bullet with "- "
       - Focus on the purpose and impact, not implementation details
       - Wrap at 72 characters

    TYPES (pick one):
    - feat: New feature (MINOR bump)
    - fix: Bug fix (PATCH bump)
    - docs: Documentation only
    - style: Code style/formatting
    - refactor: Code restructure (no fix/feat)
    - perf: Performance improvement
    - test: Add/update tests
    - build: Build system, Cargo.toml, dependencies
    - ci: CI/CD configuration
    - chore: Maintenance, tooling, configs
    - revert: Reverts a previous commit
    - security: Security fixes/hardening

    SCOPES (pick one, or omit if changes span multiple):
    - core, shared, domain (Core layer)
    - ports (Ports layer)
    - app, config, api (Application layer)
    - adapters (Adapters layer)
    - infra, facade (Infrastructure layer)
    - testkit, cli (Testing & Binaries)
    - deps, tooling, ci, docker, docs, tests, scripts (Cross-cutting)

    RULES:
    1. Analyze the diff to determine the most appropriate type and scope
    2. Write a concise subject line summarizing the change
    3. Write body bullets that explain what was done and why it matters
    4. Output ONLY the formatted commit message (subject + body), nothing else

    EXAMPLE OUTPUT:
    feat(app): implement codebase indexing use case

    - Added index_codebase function to orchestrate scanning, splitting, and embedding
    - Introduced in-memory adapters for embedding and vector DB for testing
    - Implemented progress tracking and cancellation support during indexing
    - Created integration tests ensuring deterministic behavior

    DIFF:
    PROMPT_EOF
    )

    # Call Claude CLI
    echo "🤖 Generating commit message..."
    MSG=$(echo "$DIFF" | claude --print "$PROMPT")

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "$MSG"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📋 Copied to clipboard (macOS)"
    echo "$MSG" | pbcopy

# Generate AI commit message and commit directly
commit-ai: commit-msg
    #!/usr/bin/env bash
    set -euo pipefail
    MSG=$(pbpaste)
    echo ""
    read -p "Commit with this message? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git commit -m "$MSG"
        echo "✓ Committed!"
    else
        echo "Aborted. Message still in clipboard."
    fi

# =============================================================================
# HELP
# =============================================================================

# Show detailed help
help:
    @echo ""
    @echo "semantic-code-agents-rs - Development Tasks"
    @echo "============================================"
    @echo ""
    @echo "Pre-commit Hooks:"
    @echo "  just pc        # Run hooks on staged files"
    @echo "  just pc-all    # Run hooks on all files"
    @echo "  just pc-push   # Run pre-push hooks on all files"
    @echo "  just pc-full   # Stage + pre-commit + pre-push (full gate)"
    @echo "  just pc-install # Install all hook types"
    @echo "  just pc-update # Update hooks to latest versions"
    @echo ""
    @echo "Git / Commits:"
    @echo "  just commit-msg # Generate AI commit message (copies to clipboard)"
    @echo "  just commit-ai  # Generate + commit interactively"
    @echo ""
    @echo "Before Committing:"
    @echo "  just pc-all    # Run all pre-commit hooks"
    @echo ""
    @echo "Before Pushing:"
    @echo "  just pc-full   # Full pre-commit + pre-push gate"
    @echo ""
    @echo "Run 'just --list' for all available commands"
    @echo ""
