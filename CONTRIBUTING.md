# Contributing to microbiomeML

Thank you for your interest in contributing to microbiomeML! This document provides guidelines and instructions for contributing to the project.

## Development Setup

This project uses [Pixi](https://pixi.sh) for dependency management and task automation.

### Prerequisites

1. Install Pixi:
   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/thepatientwait/microbiomeML.git
   cd microbiomeML
   ```

3. Install dependencies:
   ```bash
   pixi install
   ```

### Development Tasks

The project provides several Pixi tasks to help with development:

#### Code Formatting
```bash
# Run all formatters (autoflake, docformatter, isort, black)
pixi run format

# Or run individual formatters:
pixi run format-autoflake     # Remove unused imports/variables
pixi run format-docformatter  # Format docstrings
pixi run format-isort         # Sort imports
pixi run format-black         # Format code with Black
```

#### Code Quality Checks
```bash
# Run linter (flake8)
pixi run lint

# Run type checker (mypy)
pixi run type-check
```

#### Testing
```bash
# Run all tests
pixi run test

# Run tests with coverage report
pixi run test-cov
```

#### Pre-commit Check
```bash
# Run all checks (lint, type-check, test) before committing
pixi run local-check
```

## Code Style Guidelines

### Configuration Location

All code style settings are centralized in **[`pyproject.toml`](pyproject.toml)** to ensure consistency across all tools:

- **Black** formatting rules: `[tool.black]`
- **isort** import sorting: `[tool.isort]`
- **mypy** type checking: `[tool.mypy]`
- **flake8** linting: `[tool.flake8]` (via flake8-pyproject plugin)

Both Pixi tasks and pre-commit hooks read from these configuration files, so you only need to update settings in one place.

### Python Style

- **Line length**: 79 characters (configured in [`pyproject.toml`](pyproject.toml))
- **Import sorting**: Uses `isort` with Black profile (configured in [`pyproject.toml`](pyproject.toml))
- **Code formatting**: Uses `black` formatter (configured in [`pyproject.toml`](pyproject.toml))
- **Docstring style**: Google-style docstrings, wrapped at 79 characters

### Type Hints

- All function signatures should include type hints
- Use `Optional[T]` for nullable types
- Use `Union[A, B]` for multi-type parameters
- Type checking is enforced with `mypy`

### Testing

- All new features must include tests
- Tests are located in `test/tests/`
- Use pytest with assert statements (no unittest-style assertions)
- Test fixtures are defined in `test/tests/conftest.py`
- Aim for >80% code coverage

### Test Organization

Tests are organized by component:
- `test_metadata.py` - SampleMetadata tests
- `test_profiles.py` - TaxonomicProfiles tests
- `test_features.py` - FeatureSet tests
- `test_dataset.py` - Dataset tests
- `test_integration.py` - Integration and workflow tests

## Git Workflow

### Branches

- `main` - Stable production code
- `develop` - Development branch for integration
- Feature branches: `feature/your-feature-name`
- Bug fixes: `fix/issue-description`

### Commit Messages

Follow conventional commit format:
```
<type>: <description>

[optional body]

[optional footer]
```

Types:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### Pre-commit Hooks

The project uses pre-commit hooks to ensure code quality. Install them with:

```bash
pixi run pre-commit install
```

This will automatically run formatters and linters before each commit.

## Pull Request Process

1. **Create a feature branch** from `develop`:
   ```bash
   git checkout develop
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the style guidelines

3. **Run the local check** to ensure everything passes:
   ```bash
   pixi run local-check
   ```

4. **Format your code**:
   ```bash
   pixi run format
   ```

5. **Commit your changes** with a descriptive commit message

6. **Push to your fork** and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Ensure CI passes** - GitHub Actions will automatically run tests on your PR

8. **Request a review** from maintainers

### PR Checklist

- [ ] Code follows the project's style guidelines
- [ ] All tests pass (`pixi run test`)
- [ ] New features include tests
- [ ] Documentation has been updated if needed
- [ ] Commit messages follow conventional commit format
- [ ] `pixi run local-check` passes without errors

## Continuous Integration

GitHub Actions automatically runs on all pushes and pull requests:
- Linting with flake8
- Type checking with mypy
- Full test suite with pytest
- Coverage reporting

All checks must pass before a PR can be merged.

## Project Structure

```
microbiomeML/
├── src/
│   └── microbiome_ml/
│       ├── core/         # Core configuration
│       ├── utils/        # Utility functions and helpers
│       ├── wrangle/      # Data wrangling (Dataset, Metadata, Profiles, Features)
│       ├── train/        # Model training (future)
│       └── visualise/    # Visualization tools (future)
├── test/
│   ├── data/             # Test data files
│   └── tests/            # Test suite
│       ├── conftest.py   # Shared fixtures
│       └── test_*.py     # Test modules
├── .github/
│   └── workflows/        # CI/CD workflows
└── pixi.toml             # Pixi configuration and tasks
```

## Key Components

- **Dataset**: Main entry point for working with microbiome data
- **SampleMetadata**: Sample metadata and attributes management
- **TaxonomicProfiles**: Taxonomic abundance profiles
- **FeatureSet**: ML-ready feature matrices

All components support both eager (in-memory) and lazy (streaming) modes.

Thank you for contributing! 🎉
