# Documentation Overview

This document provides an overview of the batcharray documentation structure and how to navigate it.

## Documentation Website

The main documentation is hosted at: [https://durandtibo.github.io/batcharray/](https://durandtibo.github.io/batcharray/)

## Documentation Structure

### Getting Started

- **[README.md](README.md)**: Quick introduction and installation instructions
- **[Get Started](docs/docs/get_started.md)**: Detailed installation guide with multiple installation methods
- **[CHANGELOG.md](CHANGELOG.md)**: Version history and release notes

### User Guides

Comprehensive guides for each module with examples:

1. **[Array Operations](docs/docs/uguide/array.md)**: Working with single NumPy arrays
   - Batch operations (slicing, splitting, statistics)
   - Sequence operations (time series, variable length sequences)
   - Sorting and permutation
   - Working with masked arrays

2. **[Nested Structures](docs/docs/uguide/nested.md)**: Working with dictionaries and lists of arrays
   - Manipulating complex data structures
   - Maintaining relationships between arrays
   - Mathematical operations on nested data
   - Shuffling and permutation while preserving relationships

3. **[Computation Models](docs/docs/uguide/computation.md)**: Abstracting array operations
   - Automatic type selection
   - Working with regular and masked arrays
   - Creating custom computation models

4. **[Recursive Operations](docs/docs/uguide/recursive.md)**: Traversing nested structures
   - Applying functions recursively
   - Custom transformations
   - Type-specific operations

5. **[Utility Functions](docs/docs/uguide/utils.md)**: Exploring and validating data
   - Breadth-first search (BFS)
   - Depth-first search (DFS)
   - Structure validation and debugging

### Practical Resources

- **[Examples](examples/)**: Working code examples demonstrating real-world usage
  - Basic array operations
  - Nested data structures
  - Machine learning pipelines
  
- **[FAQ](docs/docs/faq.md)**: Frequently asked questions and answers
  - General questions about batcharray
  - Usage questions and common patterns
  - Performance and troubleshooting

- **[Troubleshooting Guide](docs/docs/troubleshooting.md)**: Solutions to common problems
  - Error messages and fixes
  - Performance optimization
  - Data validation issues

- **[Migration Guide](docs/docs/migration.md)**: Upgrading between versions
  - Version-specific changes
  - API compatibility notes
  - Testing migration

### API Reference

Complete API documentation for all modules:

- **[array](docs/docs/refs/array.md)**: Array module reference
- **[nested](docs/docs/refs/nested.md)**: Nested module reference
- **[computation](docs/docs/refs/computation.md)**: Computation module reference
- **[recursive](docs/docs/refs/recursive.md)**: Recursive module reference
- **[utils](docs/docs/refs/utils.md)**: Utils module reference

### Contributing

- **[CONTRIBUTING.md](.github/CONTRIBUTING.md)**: Guide for contributors
  - Development environment setup
  - Code style guidelines
  - Testing requirements
  - Pull request process

- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)**: Community guidelines

## Quick Links by Use Case

### I want to...

**Learn batcharray basics:**
1. Read the [README](README.md)
2. Follow the [Get Started](docs/docs/get_started.md) guide
3. Try the [basic examples](examples/basic_array_operations.py)

**Work with batches of data:**
1. Read [Array Operations Guide](docs/docs/uguide/array.md)
2. Try [basic_array_operations.py](examples/basic_array_operations.py)

**Work with complex nested data:**
1. Read [Nested Structures Guide](docs/docs/uguide/nested.md)
2. Try [nested_data_structures.py](examples/nested_data_structures.py)

**Build a machine learning pipeline:**
1. Read [Nested Structures Guide](docs/docs/uguide/nested.md)
2. Try [ml_data_pipeline.py](examples/ml_data_pipeline.py)

**Handle missing data:**
1. Read "Working with Masked Arrays" in [Array Operations](docs/docs/uguide/array.md)
2. Check [FAQ](docs/docs/faq.md) for masked array questions

**Solve a problem:**
1. Check the [FAQ](docs/docs/faq.md)
2. Read the [Troubleshooting Guide](docs/docs/troubleshooting.md)
3. Search [GitHub Issues](https://github.com/durandtibo/batcharray/issues)

**Upgrade to a new version:**
1. Read the [CHANGELOG](CHANGELOG.md)
2. Follow the [Migration Guide](docs/docs/migration.md)

**Contribute to batcharray:**
1. Read [CONTRIBUTING.md](.github/CONTRIBUTING.md)
2. Check existing [issues](https://github.com/durandtibo/batcharray/issues)
3. Follow the development setup instructions

## Building Documentation Locally

To build and view the documentation locally:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build and serve documentation
mkdocs serve

# View at http://127.0.0.1:8000/
```

For development:

```bash
# Install all dependencies including docs
make install-all

# Serve documentation with live reload
cd docs
mkdocs serve
```

## Documentation Sources

- **Markdown files**: `docs/docs/*.md` - Main documentation content
- **User guides**: `docs/docs/uguide/*.md` - Detailed module guides
- **API references**: `docs/docs/refs/*.md` - Auto-generated from docstrings
- **Examples**: `examples/*.py` - Working code examples
- **Configuration**: `docs/mkdocs.yml` - MkDocs configuration

## Contributing to Documentation

We welcome documentation contributions! You can help by:

1. **Fixing typos or errors**: Submit a PR with corrections
2. **Adding examples**: Create new example scripts in `examples/`
3. **Improving guides**: Enhance existing user guides with more details
4. **Adding FAQ entries**: Share common questions and solutions
5. **Writing tutorials**: Create step-by-step tutorials for specific use cases

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for more details on contributing.

## Getting Help

If you can't find what you're looking for:

1. Search the documentation using the search box
2. Check the [FAQ](docs/docs/faq.md)
3. Look through [GitHub Issues](https://github.com/durandtibo/batcharray/issues)
4. Ask a question by [opening an issue](https://github.com/durandtibo/batcharray/issues/new)

## Documentation License

The documentation is licensed under the same terms as the code (BSD 3-Clause License).
See [LICENSE](LICENSE) for details.
