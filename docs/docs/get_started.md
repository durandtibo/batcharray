# Get Started

It is highly recommended to install in
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
to keep your system in order.

## Installing with `uv pip` (recommended)

The following command installs the latest version of the library:

```shell
uv pip install batcharray
```

To make the package as slim as possible, only the packages required to use `batcharray` are
installed.
It is possible to install all the optional dependencies by running the following command:

```shell
uv pip install 'batcharray[all]'
```

## Installing from source

To install `batcharray` from source for development purposes, please refer to the development setup instructions in the [CONTRIBUTING.md](https://github.com/durandtibo/batcharray/blob/main/CONTRIBUTING.md) file in the repository.
