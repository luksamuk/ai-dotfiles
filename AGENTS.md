# AI Agents Repository

This repository contains scripts and configuration files for managing local AI models using Ollama and setting up AI tools.

## Build Commands

```bash
# Run the ask-ai script
python3 scripts/ask-ai.py

# Example usage with different models
python3 scripts/ask-ai.py -m qwen "What is the capital of France?"

# Enable think mode
python3 scripts/ask-ai.py -t -m glm "Explain quantum computing"

# Debug mode
python3 scripts/ask-ai.py -d -m gpt "Debug information please"
```

## Lint Commands

```bash
# Lint Python scripts
flake8 scripts/ask-ai.py

# Check Python code style
pylint scripts/ask-ai.py

# For shell scripts (if any)
shellcheck scripts/*
```

## Documentation Verification

When working with configuration files, agents should:
- Verify syntax compatibility with respective tools (ollama, opencode, etc.)
- Check that configuration parameters match documented standards
- Ensure file structure follows documented conventions
- Validate that referenced models and paths exist and are accessible

All documentation should be verified against official tool documentation available online, not through automated testing.

## Lint Commands

```bash
# Lint Python scripts
flake8 scripts/ask-ai.py

# Check Python code style
pylint scripts/ask-ai.py

# For shell scripts (if any)
shellcheck scripts/*
```



## Code Style Guidelines

1. **Python Style**
   - Use PEP 8 style guide
   - Indentation: 4 spaces
   - Maximum line length: 88 characters
   - Use descriptive variable names
   - Follow function and class naming conventions
   - Add docstrings for functions

2. **Documentation**
   - Add comments explaining complex logic
   - Include usage examples in docstrings
   - Maintain README updates
   - Follow project structure conventions

3. **Security**
   - Never commit credential files
   - Ensure proper input validation
   - Sanitize user inputs
   - Use subprocess safely

4. **Repository Structure**
   - Keep scripts in scripts/ directory
   - Use clear and descriptive file names
   - Maintain consistent folder structure
   - Follow semantic versioning for scripts

## Cursor/Copilot Rules

### Cursor

1. **Model Selection**
   - When working with AI models, prefer using the specific model parameters as defined in the script
   - Ensure model selection matches available local models (gpt, qwen, glm, lfm, etc.)
   - Consider model capabilities when writing code

2. **Think Mode Configuration**
   - Enable think mode using `-t` flag for appropriate models
   - Verify that target models support think capability
   - When using think mode, expect longer response times

3. **Code Generation**
   - When generating code, ensure it respects the script requirements and parameter handling
   - Consider Ollama's command-line API limitations
   - Write output that works with the existing Markdown formatting pipeline

4. **Python Code Standards**
   - Follow PEP 8 guidelines
   - Use meaningful variable names and comments
   - Include error handling for subprocess calls
   - Respect Python version compatibility (Python 3.6+)

### Copilot

1. **Model Specificity**
   - Copilot should be aware of the specific models available in this project
   - When suggesting code changes, consider compatibility with the supported models (gpt, qwen, glm, lfm, etc.)
   - Remember that this repository uses Ollama CLI with specific model mappings

2. **Integration Patterns**
   - The script is designed to interface with the Ollama CLI
   - Any code changes should maintain compatibility with the existing subprocess API approach
   - The system supports both command-line arguments and piped input

3. **Error Handling**
   - Maintain existing error handling patterns for missing models
   - Preserve existing debug functionality for troubleshooting
   - Consider edge cases in Ollama CLI interaction

## Project Information

The primary script in this repository is `scripts/ask-ai.py` which provides a command-line interface for querying local AI models through Ollama. It supports multiple models including:

- GPT-OSS (gpt)
- Qwen3-Coder (qwen) 
- GLM 4.7 Flash (glm)
- LFM 2.5 Thinking (lfm)
- And other local models

The script is designed for quick question answering through command line, with support for:
- Model switching
- Think mode activation
- Debug output
- Piped input handling

## Usage Examples

```bash
# Ask a question with default model
python3 scripts/ask-ai.py "What is machine learning?"

# Ask with a specific model
python3 scripts/ask-ai.py -m qwen "Explain quantum computing"

# Enable thinking mode
python3 scripts/ask-ai.py -t -m glm "Analyze this problem"

# See help
python3 scripts/ask-ai.py -h

# Debug mode
python3 scripts/ask-ai.py -d -m gpt "Debug me"
```

## Environment Requirements

- Python 3.6 or higher
- Ollama CLI installed and configured
- Available local AI models (see MODEL_MAP in script)
- `glow` for Markdown rendering (optional but recommended)