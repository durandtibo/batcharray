# Migration Guide

This guide helps you migrate your code between different versions of `batcharray`.

## Migrating to 0.2.x

### Overview

Version 0.2.x introduces several changes to the build system and dependencies while maintaining API compatibility with 0.1.x. Most code should work without modifications.

### Build System Changes

**From Poetry to uv/hatchling**

The project has migrated from Poetry to using `uv` for dependency management and `hatchling` for building.

**What this means for users:**

- **Installing from PyPI**: No changes needed, continue using `pip install batcharray`
- **Installing from source**: Follow the updated instructions in [Get Started](get_started.md#installing-from-source)
- **Contributing**: See updated [CONTRIBUTING.md](.github/CONTRIBUTING.md) for new development setup

### Python Version Support

**Minimum Python version increased from 3.9 to 3.10**

If you're using Python 3.9:

1. **Option 1**: Upgrade to Python 3.10 or later (recommended)
2. **Option 2**: Continue using batcharray 0.1.x

```bash
# Check your Python version
python --version

# If < 3.10, install old version
pip install "batcharray<0.2.0"
```

### Dependency Updates

**Updated dependencies:**

- `coola`: `>=0.8.4,<1.0` → `>=0.9.1,<1.0`
- `numpy`: Continues to support `>=1.22,<3.0`

**What you need to do:**

Most users won't need to do anything. If you have strict dependency pinning:

```bash
# Update your requirements
pip install --upgrade batcharray coola
```

### API Changes

**No breaking API changes in 0.2.x**

All functions and modules maintain backward compatibility. Your existing code should continue to work without modifications.

### Testing Your Migration

After upgrading, run your test suite:

```python
import batcharray
import numpy as np

# Verify installation
print(f"batcharray version: {batcharray.__version__}")

# Test basic functionality
from batcharray import array, nested

# Test array operations
arr = np.array([[1, 2], [3, 4]])
result = array.slice_along_batch(arr, stop=1)
assert result.shape == (1, 2)

# Test nested operations
data = {"a": np.array([1, 2, 3]), "b": np.array([4, 5, 6])}
sliced = nested.slice_along_batch(data, stop=2)
assert sliced["a"].shape == (2,)

print("✓ Migration successful!")
```

## Migrating from 0.0.x to 0.1.x

### Major Changes

Version 0.1.x introduced the stable API that continues in 0.2.x. Key changes from 0.0.x:

1. **Module reorganization**: Some internal modules were reorganized
2. **Computation models**: Added computation abstraction layer
3. **Enhanced documentation**: Comprehensive user guides added

### Deprecations

If you were using internal APIs (modules/functions starting with `_`), they may have changed or been removed. Only use public APIs documented in the reference.

## Migrating from Earlier Versions

### From 0.0.3 to 0.1.0

**NumPy version support expanded:**

- 0.0.3: `numpy>=1.22,<2.0`
- 0.1.0+: `numpy>=1.22,<3.0`

No code changes needed, but you can now use NumPy 2.x if desired.

### From 0.0.2 to 0.0.3

No breaking changes. Enhanced test coverage and bug fixes.

### From 0.0.1 to 0.0.2

No breaking changes. Dependency updates only.

## General Migration Tips

### 1. Check Version Requirements

Before upgrading, check your project's requirements:

```bash
# List current batcharray version
pip show batcharray

# Check if upgrade is available
pip index versions batcharray
```

### 2. Use Virtual Environments

Always test upgrades in a virtual environment first:

```bash
python -m venv test_env
source test_env/bin/activate  # Unix
# or
test_env\Scripts\activate  # Windows

pip install batcharray==0.2.0
# Run your tests
```

### 3. Pin Dependencies Appropriately

For production:

```txt
# requirements.txt
batcharray>=0.2.0,<0.3.0  # Allow patch updates
numpy>=1.22,<3.0
```

For development:

```txt
# requirements-dev.txt
batcharray==0.2.0  # Exact version
```

### 4. Update Import Statements

Always import from public modules:

```python
# ✓ Good - public API
from batcharray import array, nested, computation
from batcharray.recursive import recursive_apply

# ✗ Bad - internal APIs (may break)
from batcharray.array.slicing import _some_internal_function
```

### 5. Check Deprecation Warnings

Run with warnings enabled to catch future deprecations:

```python
import warnings

warnings.simplefilter("always", DeprecationWarning)

# Your code here
```

### 6. Review Changelog

Always review the [CHANGELOG.md](../CHANGELOG.md) for detailed information about changes in each version.

## Troubleshooting Migration Issues

### ImportError after upgrade

**Problem**: Getting import errors after upgrading

**Solution**:

```bash
# Clear pip cache
pip cache purge

# Reinstall
pip uninstall batcharray
pip install batcharray

# Or force reinstall
pip install --force-reinstall batcharray
```

### Dependency conflicts

**Problem**: Conflicts with other packages

**Solution**:

```bash
# Check for conflicts
pip check

# See dependency tree
pip install pipdeptree
pipdeptree -p batcharray

# Consider using a fresh environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install batcharray
```

### Tests failing after upgrade

**Problem**: Tests pass on old version but fail on new version

**Solution**:

1. Check if you're using internal APIs (shouldn't be)
2. Verify NumPy and other dependency versions
3. Review the changelog for subtle behavior changes
4. Report issue if API contract seems broken

```python
# Debug script
import sys
import numpy as np
import batcharray

print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"batcharray: {batcharray.__version__}")

# Run minimal failing test here
```

## Getting Help

If you encounter migration issues:

1. Check the [FAQ](faq.md) for common questions
2. Review the [Troubleshooting Guide](troubleshooting.md)
3. Search [existing issues](https://github.com/durandtibo/batcharray/issues)
4. Open a new issue with:
   - Old and new version numbers
   - Python version
   - Error messages
   - Minimal reproduction code

## Contributing to Migration Documentation

If you encounter migration issues not covered here, please:

1. Document your solution
2. Submit a PR to update this guide
3. Help other users facing similar issues

Your contributions make migrations smoother for everyone!
