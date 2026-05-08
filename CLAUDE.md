# metal-runtime: A metal-cpp-based C++ runtime for Metal GPU

This repo is an attempt to implement a Metal GPU runtime for use in GPU computing on Apple platforms.

## Code style

C++ as per clang-format, Python as per pyproject.toml.
This is a jj VCS repo, use jj commands for everything except things that don't have a jj equivalent, e.g. git submodules.
Run `uvx prek run --all-files` to apply formatters and Python type checkers.

## Notes

Please leave implementation notes for the project inside the notes/ folder.
