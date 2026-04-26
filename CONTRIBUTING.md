# Contributing to free-claude-code

First off, thank you for considering contributing to `free-claude-code`! It's people like you that make this tool great.

## How Can I Contribute?

### Reporting Bugs
Before creating bug reports, please check a list of existing issues to see if the problem has already been reported. When you are creating a bug report, please include as many details as possible:
- **Use a clear and descriptive title.**
- **Describe the exact steps which reproduce the problem.**
- **Provide logs and error messages.**
- **Include your environment details (OS, Python version, Provider).**

### Suggesting Enhancements
Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please:
- **Use a clear and descriptive title.**
- **Provide a step-by-step description of the suggested enhancement.**
- **Explain why this enhancement would be useful.**

### Your First Code Contribution
Unsure where to begin contributing? You can start by looking through these `good first issue` and `help wanted` issues.

## Development Setup

### Prerequisites
- Python 3.10 or higher.
- [uv](https://github.com/astral-sh/uv) for dependency management.

### Setup Instructions
1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/free-claude-code.git
   cd free-claude-code
   ```
3. **Install dependencies**:
   ```bash
   uv sync
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feat/your-feature-name
   ```

## Project Structure
- `api/`: API server logic.
- `providers/`: Adapters for different LLM providers (OpenAI, Groq, etc.).
- `messaging/`: Discord and Telegram bot workers.
- `core/`: Shared logic and base classes.
- `tests/`: Unit and integration tests.

## Testing and Linting
Before submitting a PR, please ensure your code passes linting and tests:

### Linting
We use `ruff` for linting:
```bash
uv run ruff check .
```

### Testing
Run the test suite using `pytest`:
```bash
uv run pytest
```

## Pull Request Process
1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports and useful file locations.
3. You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the maintainer to merge it.

## Community
Join our [Discord](https://discord.gg/your-link) for discussions!
