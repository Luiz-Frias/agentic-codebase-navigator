# Attribution

This project is derived from the **Recursive Language Models (RLM)** research and implementation.

## Original Work

### Core RLM Research

- **Title**: Recursive Language Models
- **Authors**: Alex L. Zhang, Tim Kraska, Omar Khattab
- **Affiliation**: MIT OASYS Lab
- **Paper**: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) (December 2025)
- **Blog Post**: [alexzhang13.github.io/blog/2025/rlm](https://alexzhang13.github.io/blog/2025/rlm/) (October 2025)
- **Documentation**: [alexzhang13.github.io/rlm](https://alexzhang13.github.io/rlm/)
- **Original Repository**: [github.com/alexzhang13/rlm](https://github.com/alexzhang13/rlm)
- **Original License**: MIT (Copyright 2025 Alex Zhang)

### Citation

If you use this work in research, please cite the original paper:

```bibtex
@misc{zhang2025recursivelanguagemodels,
      title={Recursive Language Models},
      author={Alex L. Zhang and Tim Kraska and Omar Khattab},
      year={2025},
      eprint={2512.24601},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24601},
}
```

## Refactoring Changes

**Refactored by**: Luiz Frias (2026)

### Architecture Migration

| Aspect | Original | Refactored |
|--------|----------|-----------|
| **Pattern** | Flat/monolithic | Hexagonal modular monolith |
| **Package layout** | `rlm/` | `src/rlm/` (src-layout) |
| **Design** | Tight coupling | Ports/adapters pattern |
| **Testing** | Basic | Comprehensive (unit/integration/e2e/packaging/performance) |

### Key Additions

- **Hexagonal architecture layers**: domain, application, infrastructure, adapters
- **Explicit ports**: `LLMPort`, `BrokerPort`, `EnvironmentPort`, `LoggerPort`
- **Enhanced configuration system**: `RLMConfig`, `LLMConfig`, `EnvironmentConfig`, `LoggerConfig`
- **TCP broker**: Request routing with wire protocol for nested execution
- **Mock LLM adapter**: Deterministic testing without API keys
- **Enhanced type safety**: Result types, frozen dataclasses, comprehensive type hints
- **Improved error handling**: Domain-specific exception hierarchy
- **CI/CD pipeline**: GitHub Actions with quality gates

### Maintained Compatibility

- Public API surface remains compatible (`RLM`, `create_rlm`, `completion`)
- Core algorithm and iteration loop behavior unchanged
- All original LLM providers supported (OpenAI, Anthropic, Gemini, etc.)
- All original environment adapters ported (Local, Docker, Modal)

## License

MIT License - See [LICENSE](LICENSE) for details.

- Original work: Copyright (c) 2025 Alex Zhang
- Refactored work: Copyright (c) 2026 Luiz Frias
