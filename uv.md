# UV

> An extremely fast Python package and project manager, written in Rust.

- Repository: https://github.com/astral-sh/uv
- Docs: https://docs.astral.sh/uv/
- Features: https://docs.astral.sh/uv/getting-started/features/

Why UV? Python is easy to use, but hard to distribute. You write a great script, Jupyter notebook, or package. How do you get it to someone else? With UV, they clone the repo, install UV and `uv sync` and it handling installing the correct version of Python (without interfering with your system version), all of the packages that are needed. This simplifies reproducibility substantially.

## Install UV on Linux and Mac

Reference: https://docs.astral.sh/uv/getting-started/installation/

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install UV on Windows

Reference: https://docs.astral.sh/uv/getting-started/installation/

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

```

## Shell Autocomplete

Reference: https://docs.astral.sh/uv/getting-started/installation/#shell-autocompletion

### Linux and Mac

Run one of the following:

```bash
# Determine your shell (e.g., with `echo $SHELL`), then run one of:
echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc
echo 'eval "$(uv generate-shell-completion zsh)"' >> ~/.zshrc
echo 'uv generate-shell-completion fish | source' >> ~/.config/fish/config.fish
echo 'eval (uv generate-shell-completion elvish | slurp)' >> ~/.elvish/rc.elv
```

### Windows

```bash
Add-Content -Path $PROFILE -Value '(& uv generate-shell-completion powershell) | Out-String | Invoke-Expression'
```

## Build the Virtual Environment and you are up and running

In the repository root, execute:

```bash
uv sync
```

That command will build the virtual environment by:

- Downloading and installing the correct version of Python.
- Download and install the correct packages.

> NOTE: This needs to be run from within the repository. If you add new dependencies or modify the `pyproject.toml` you should run `uv sync` to update the virtual environment.

## Update/Upgrade UV

Reference: https://docs.astral.sh/uv/getting-started/installation/#next-steps

To update UV to the latest version, issue the following command:

```bash
uv self update
```

## Manage Python Versions

Reference: https://docs.astral.sh/uv/guides/install-python/

**Python versions:**

- `uv python install` - Install Python versions.
- `uv python list` - View available Python versions.
- `uv python find` - Find an installed Python version.
- `uv python pin` - Pin the current project to use a specific Python version.
- `uv python uninstall` - Uninstall a Python version.

## Run Python scripts (PY files)

Reference: https://docs.astral.sh/uv/guides/scripts/

Executing standalone Python scripts, e.g., `example.py`.

- `uv run` - Run a script.
- `uv add --script` - Add a dependency to a script
- `uv remove --script` - Remove a dependency from a script

## Projects

Reference: https://docs.astral.sh/uv/guides/projects/

Creating and working on Python projects, i.e., with a `pyproject.toml`.

- `uv init` -  Create a new Python project.
- `uv add` -  Add a dependency to the project.
- `uv remove` -  Remove a dependency from the project.
- `uv sync` -  Sync the project's dependencies with the environment.
- `uv lock` -  Create a lockfile for the project's dependencies.
- `uv run` -  Run a command in the project environment.
- `uv tree` -  View the dependency tree for the project.

## Tools

Reference: https://docs.astral.sh/uv/guides/tools/

Running and installing tools published to Python package indexes, e.g., **ruff** or **black**.

- `uvx / uv tool run` - Run a tool in a temporary environment.
- `uv tool install` - Install a tool user-wide.
- `uv tool uninstall` - Uninstall a tool.
- `uv tool list` - List installed tools.
- `uv tool update-shell` - Update the shell to include tool executables.

> NOTE: `uvx ruff` is a shortcut for `uv tool run ruff`.

## Utility

Managing and inspecting uv's state, such as the cache, storage directories, or performing a self-update:

- `uv cache clean` - Remove cache entries.
- `uv cache prune` - Remove outdated cache entries.
- `uv cache dir` - Show the uv cache directory path.
- `uv tool dir` - Show the uv tool directory path.
- `uv python dir` - Show the uv installed Python versions path.
- `uv self update` - Update uv to the latest version.
