# Home

<p align="center">
    <a href="https://github.com/durandtibo/batcharray/actions">
        <img alt="CI" src="https://github.com/durandtibo/batcharray/workflows/CI/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/batcharray/actions">
        <img alt="Nightly Tests" src="https://github.com/durandtibo/batcharray/workflows/Nightly%20Tests/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/batcharray/actions">
        <img alt="Nightly Package Tests" src="https://github.com/durandtibo/batcharray/workflows/Nightly%20Package%20Tests/badge.svg">
    </a>
    <br/>
    <a href="https://durandtibo.github.io/batcharray/">
        <img alt="Documentation" src="https://github.com/durandtibo/batcharray/workflows/Documentation%20(stable)/badge.svg">
    </a>
    <a href="https://durandtibo.github.io/batcharray/">
        <img alt="Documentation" src="https://github.com/durandtibo/batcharray/workflows/Documentation%20(unstable)/badge.svg">
    </a>
    <br/>
    <a href="https://codecov.io/gh/durandtibo/batcharray">
        <img alt="Codecov" src="https://codecov.io/gh/durandtibo/batcharray/branch/main/graph/badge.svg">
    </a>
    <a href="https://codeclimate.com/github/durandtibo/batcharray/maintainability">
        <img src="https://api.codeclimate.com/v1/badges/148edc26add138d04928/maintainability" />
    </a>
    <a href="https://codeclimate.com/github/durandtibo/batcharray/test_coverage">
        <img src="https://api.codeclimate.com/v1/badges/148edc26add138d04928/test_coverage" />
    </a>
    <br/>
    <a href="https://github.com/psf/black">
        <img  alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <a href="https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/%20style-google-3666d6.svg">
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" style="max-width:100%;">
    </a>
    <a href="https://github.com/guilatrova/tryceratops">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/try%2Fexcept%20style-tryceratops%20%F0%9F%A6%96%E2%9C%A8-black">
    </a>
    <br/>
    <a href="https://pypi.org/project/batcharray/">
        <img alt="PYPI version" src="https://img.shields.io/pypi/v/batcharray">
    </a>
    <a href="https://pypi.org/project/batcharray/">
        <img alt="Python" src="https://img.shields.io/pypi/pyversions/batcharray.svg">
    </a>
    <a href="https://opensource.org/licenses/BSD-3-Clause">
        <img alt="BSD-3-Clause" src="https://img.shields.io/pypi/l/batcharray">
    </a>
    <br/>
    <a href="https://pepy.tech/project/batcharray">
        <img  alt="Downloads" src="https://static.pepy.tech/badge/batcharray">
    </a>
    <a href="https://pepy.tech/project/batcharray">
        <img  alt="Monthly downloads" src="https://static.pepy.tech/badge/batcharray/month">
    </a>
    <br/>
</p>

## Overview

`batcharray` is lightweight library built on top of [NumPy](https://numpy.org/doc/stable/index.html)
to manipulate nested data structure with NumPy arrays.
This library provides functions for arrays where the first dimension is the batch dimension.
It also provides functions for arrays representing a batch of sequences where the first dimension
is the batch dimension and the second dimension is the sequence dimension.

## API stability

:warning: While `batcharray` is in development stage, no API is guaranteed to be stable from one
release to the next.
In fact, it is very likely that the API will change multiple times before a stable 1.0.0 release.
In practice, this means that upgrading `batcharray` to a new version will possibly break any code
that was using the old version of `batcharray`.

## License

`batcharray` is licensed under BSD 3-Clause "New" or "Revised" license available
in [LICENSE](LICENSE) file.
