# Contributing to `batcharray`

We want to make contributing to this project as easy and transparent as possible.

## Overview

We welcome contributions from anyone, even if you are new to open source.

- If you are planning to contribute back bug-fixes, please do so without any further discussion.
- If you plan to contribute new features, utility functions, or extensions to the core, please first
  open an issue and discuss the feature with us.

Once you implement and test your feature or bug-fix, please submit a Pull Request.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. Set up your development environment (see [Installation](#installation) below).
3. If you've added code that should be tested, add tests.
4. If you've changed APIs, update the documentation.
5. Ensure the test suite passes. You can use the following command to run the tests:
   ```shell
   invoke unit-test --cov
   ```
6. Make sure your code lints. The following commands can help you to format the code:
   ```shell
   pre-commit run --all-files
   ```

## Installation

### Prerequisites

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.
Please refer to the [uv installation documentation](https://docs.astral.sh/uv/getting-started/installation/) to install it.

### Setting up the development environment

1. Clone your fork:
   ```shell
   git clone git@github.com:YOUR_USERNAME/batcharray.git
   cd batcharray
   ```

2. Set up the virtual environment and install dependencies:
   ```shell
   make setup-venv
   source .venv/bin/activate
   ```

3. Install pre-commit hooks:
   ```shell
   pre-commit install
   ```

### Available invoke commands

- `inv install --no-optional-deps` - Install only core dependencies
- `inv install --docs-deps` - Install all dependencies including docs
- `inv check-lint` - Check code with ruff
- `inv check-format` - Check code formatting with black
- `inv unit-test` - Run unit tests
- `inv unit-test --cov` - Run unit tests with coverage
- `inv doctest-src` - Run doctests in source code

## Issues

We use GitHub issues to track public bugs or feature requests.
For bugs, please ensure your description is clear and concise description, and has sufficient
information to be easily reproducible.
For feature request, please add a clear and concise description of the feature proposal.
Please outline the motivation for the proposal.

## License

By contributing to `batcharray`, you agree that your contributions will be licensed under the LICENSE
file in the root directory of this source tree.
