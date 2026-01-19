# CLI Reference

RLM provides a command-line interface for quick completions and testing.

## Installation

The CLI is installed automatically with the package:

```bash
pip install agentic-codebase-navigator
```

## Basic Usage

```bash
# Simple completion
rlm completion "What is 2 + 2?"

# With prompt from stdin
echo "Explain Python decorators" | rlm completion -

# Output as JSON
rlm completion "Hello" --json
```

## Commands

### `rlm completion`

Run an RLM completion with the specified prompt.

```bash
rlm completion [OPTIONS] PROMPT
```

#### Arguments

| Argument | Description |
|----------|-------------|
| `PROMPT` | The prompt to send to the LLM. Use `-` to read from stdin. |

#### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | `openai` | LLM backend to use (`openai`, `anthropic`, `gemini`, `mock`, etc.) |
| `--model-name` | `gpt-5-nano` | Model name for the backend |
| `--environment` | `docker` | Execution environment (`local`, `docker`, `modal`, `prime`) |
| `--max-iterations` | `30` | Maximum orchestrator loop iterations |
| `--max-depth` | `1` | Maximum recursion depth for nested calls |
| `--jsonl-log-dir` | `None` | Directory for JSONL logging (enables logging if set) |
| `--json` | `False` | Output full ChatCompletion as JSON instead of just response |
| `--help` | | Show help message |

### `rlm --version`

Display the installed version:

```bash
rlm --version
# Output: rlm 1.1.0
```

### `rlm --help`

Display help information:

```bash
rlm --help
```

## Examples

### Basic Completions

```bash
# Simple question
rlm completion "What is the capital of Japan?"

# Math problem (uses code execution)
rlm completion "Calculate the factorial of 10"

# Code generation
rlm completion "Write a Python function to check if a number is prime"
```

### Backend Selection

```bash
# OpenAI (default)
rlm completion "Hello" --backend openai --model-name gpt-4

# Anthropic
rlm completion "Hello" --backend anthropic --model-name claude-3-opus-20240229

# Google Gemini
rlm completion "Hello" --backend gemini --model-name gemini-pro

# Mock (for testing, no API calls)
rlm completion "Hello" --backend mock
```

### Environment Selection

```bash
# Local execution (in-process Python)
rlm completion "Calculate 2+2" --environment local

# Docker execution (isolated container)
rlm completion "Run risky code" --environment docker
```

### JSON Output

```bash
# Full ChatCompletion object as JSON
rlm completion "What is 2+2?" --json

# Example output:
# {
#   "root_model": "gpt-4",
#   "prompt": "What is 2+2?",
#   "response": "4",
#   "usage_summary": {...},
#   "execution_time": 1.234
# }
```

### Logging

```bash
# Enable JSONL logging to a directory
rlm completion "Complex task" --jsonl-log-dir ./logs

# Logs are written to: ./logs/rlm_run_<timestamp>.jsonl
```

### Iteration Control

```bash
# Limit iterations (for complex tasks)
rlm completion "Solve this step by step..." --max-iterations 50

# Allow nested recursion
rlm completion "Deep analysis task" --max-depth 3
```

## Environment Variables

The CLI respects these environment variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google Gemini API key |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `PORTKEY_API_KEY` | Portkey API key |

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Error (invalid arguments, API error, execution failure) |

## Scripting Examples

### Batch Processing

```bash
#!/bin/bash
# Process multiple prompts from a file

while IFS= read -r prompt; do
    echo "Processing: $prompt"
    rlm completion "$prompt" --json >> results.jsonl
done < prompts.txt
```

### Pipeline Integration

```bash
# Generate code and save to file
rlm completion "Write a Python hello world script" > hello.py

# Chain with other tools
cat data.csv | rlm completion "Analyze this CSV data: $(cat -)" --json | jq '.response'
```

### CI/CD Usage

```bash
# Verify RLM is working in CI
rlm completion "Return exactly: OK" --backend mock --json | jq -e '.response == "OK"'
```

## Troubleshooting

### "Command not found: rlm"

Ensure the package is installed and your PATH includes the Python bin directory:

```bash
# Check installation
pip show agentic-codebase-navigator

# Reinstall if needed
pip install --force-reinstall agentic-codebase-navigator
```

### "API key not configured"

Set the appropriate environment variable:

```bash
export OPENAI_API_KEY="sk-..."
```

### "Docker not available"

For `--environment docker`, ensure Docker is running:

```bash
docker info
# If error, start Docker daemon
```

Use `--environment local` if Docker is not available.

## See Also

- [Getting Started](getting-started.md) — Quick start guide
- [Configuration](configuration.md) — All configuration options
- [LLM Providers](providers/llm-providers.md) — Provider setup
- [Troubleshooting](troubleshooting.md) — Common issues
