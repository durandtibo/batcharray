# Get Started

It is highly recommended to install in
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to keep your system in order.

## Installing with `pip` (recommended)

The following command installs the latest version of the library:

```shell
pip install batcharray
```

To make the package as slim as possible, only the packages required to use `batcharray` are
installed.
It is possible to install all the optional dependencies by running the following command:

```shell
pip install 'batcharray[all]'
```

## Installing from source

To install `batcharray` from source, you can follow the steps below.

### Prerequisites

The project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. First, install `uv` if you haven't already:

```shell
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Using pip
pip install uv
```

You can verify the installation by running:

```shell
uv --version
```

### Clone the repository

Clone the git repository:

```shell
git clone git@github.com:durandtibo/batcharray.git
cd batcharray
```

### Set up development environment

Create a Python 3.10+ virtual environment and install dependencies:

```shell
make setup-venv
```

This command will:
1. Update `uv` to the latest version
2. Create a virtual environment with Python 3.13
3. Install `invoke` for task management
4. Install all dependencies including documentation dependencies

Alternatively, you can create a conda environment:

```shell
make conda
conda activate batcharray
make install
```

### Install the package

To install only the core dependencies:

```shell
make install
```

To install all dependencies including documentation tools:

```shell
make install-all
```

### Verify installation

Run the test suite to verify everything is working correctly:

```shell
make unit-test-cov
```
