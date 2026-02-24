# Contributing to Bulldozer

Thank you for your interest in contributing to **Bulldozer**!  
We welcome all kinds of contributions: bug reports, documentation improvements, bug fixes, and new features.

This document describes how to contribute effectively and consistently to the project.

---

## Table of contents

1. [Reporting issues](#reporting-issues)
2. [Development setup](#development-setup)
3. [Contributing workflow](#contributing-workflow)
4. [Code style & tooling](#code-style--tooling)
5. [Coding guidelines](#coding-guidelines)
6. [Pull request acceptance process](#pull-request-acceptance-process)

---

## Reporting issues

Any proven or suspected malfunction should be reported by opening an issue in the **Bulldozer**
[GitHub repository](https://github.com/CNES/bulldozer).

**Do not hesitate to report issues.**  
It is always better to open a bug report early than to let a problem persist.  
Reporting bugs is a valuable way to contribute to the project.

When reporting an issue, please include as much detail as possible:
- The procedure used to set up the environment
- The command line or Python function involved
- The relevant output logs or error messages

---

## Development setup

Clone the repository and install the development environment:

```bash
git clone https://github.com/CNES/bulldozer
cd bulldozer
make install
```

This command will:
- Create the `bulldozer_venv` virtual environment
- Install **Bulldozer** in editable mode with development and documentation dependencies
- Install and configure pre-commit hooks

You can activate the environment manually with:
```bash
source bulldozer_venv/bin/activate
```

Common development commands:
```bash
make format      # format code
make lint        # run linters
make test        # run tests
make check       # run lint + tests (CI equivalent)
make docs        # build documentation
```
---

## Contributing workflow

Any code modification must be submitted through a **Pull Request (PR)**.
Direct pushes to the `master` branch are not allowed, as this branch is protected.

We recommend opening your Pull Request as early as possible to make ongoing work visible.
If the work is not finished yet, prefix the MR title with `WIP:`.

Each Pull Request must include:
- A short and clear description of the proposed changes
- A reference to the related issue, if applicable, using `Closes #xx`

When working on a branch (strongly recommended), prefix the branch name with the issue number:
`xx-short-description`.

### Standard Bulldozer workflow

- Create an issue (or start from an existing one)
- Create a branch named `xx-issue-name`
- Open a Pull Request linked to the issue:
  - use `WIP:` in the title if needed
  - add `Closes #xx` in the description
- Develop locally
- If you use Cython, name generated C++ files as: `c_<filename>.[cpp|h]`
- Commit and push your changes (use [Conventional commits](https://www.conventionalcommits.org/) for your messages)
- Ensure all tests pass
- When ready, remove `WIP:` and request a review

---

## Code style & tooling

Bulldozer uses a set of standard Python tools to ensure code quality and consistency.
All contributors are expected to follow these rules.

### Pre-commit hooks (mandatory)

The project uses **pre-commit** to automatically run formatting, linting, and type checks.

Hooks are installed automatically when running:

```bash
make install
```

Pre-commit runs automatically before each commit.  
**Commits that do not pass pre-commit checks will be rejected.**

You can manually run all checks with:
```bash
make lint
```

Or auto-fix formatting and lint issues with:
```bash
make ruff-fix
```

---

### Code formatting

Bulldozer uses **Ruff** as a unified tool for:
- Code formatting
- Import sorting
- Linting
- Modern Python syntax upgrades

Ruff replaces multiple legacy tools (Black, isort, flake8, pylint).  

Configuration is centralized in `pyproject.toml`.

Key rules:
- Line length: **120**
- Imports are automatically sorted
- Code is automatically formatted
- Common Python issues and potential bugs are checked

You should not manually fight formatting changes, let Ruff handle it.

---


### Static typing

- **mypy** is used for static type checking
- Type hints are required for all new functions and public APIs
- Existing code may be progressively typed, but new code must be typed

Type checking is automatically enforced via pre-commit.

--- 

### Tests

- **pytest** is used as the test framework
- New features and bug fixes must include appropriate tests
- Tests should cover both normal and degraded cases when possible
- All tests must pass before submitting a Pull Request:
```bash
make check
```

---

## Coding guidelines

Please follow these rules when contributing code:
- Use explicit and meaningful variable and function names
- Add comments when introducing non-trivial logic
- Do **not** use `print()`. Use the Bulldozer internal logger
- All functions must be documented (purpose, parameters, return values)
- Avoid adding new dependencies unless strictly necessary and permissively licensed
- Update the documentation if necessary

---

## Pull request acceptance process

Pull Requests are reviewed by the **Bulldozer steering committee.**  

A Pull Request (PR) may be merged when:
- The code has been reviewed and approved
- All automated checks pass
- The proposed changes are consistent with the project’s scope and quality standards

Only members of the steering committee are authorized to merge into the `master` branch.