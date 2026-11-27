# Installation and Run

## Install
```bash
##DIRECT INSTALL
pip install "git+https://github.com/ZieganWesley/Private_Agent.git@main"

##SOURCE INSTALL
git clone https://github.com/ZieganWesley/Private_Agent.git
cd Private_Agent
pip install -e .
```

## Run
```bash
private-agent
```

## Diagnostics
```bash
private-agent --doctor
```

## UI Check (developer only)
```bash
python scripts/ui_check.py
```
Remove this entry point before production.

## Platform setup

### WSL (Windows)
- Install inside WSL, not on the Windows host. Keep projects on the Linux
  filesystem (e.g. `~/projects`); avoid `/mnt/c`, where filesystem calls are
  slow and file watching is unreliable.
- Bubblewrap is required for sandboxed code tasks:
  ```bash
  sudo apt-get update && sudo apt-get install -y bubblewrap
  ```
- Ollama can run either inside WSL or on the Windows host. If it runs on the
  Windows host, point Private Agent at `http://host.docker.internal:11434`
  (or the host IP from `ip route`); if it runs inside WSL, the default
  `http://localhost:11434` works.
- GPU access from WSL needs the Windows NVIDIA driver plus the WSL CUDA
  runtime; CPU-only Ollama works out of the box.

### macOS
- Install build/runtime dependencies with Homebrew:
  ```bash
  brew install python@3.14
  ```
  Use the Homebrew Python (or a pyenv build) so native extensions compile
  cleanly; the system Python is not supported.
- Sandbox code tasks use the macOS sandbox profile instead of bubblewrap, so
  no extra package is needed.
- If `private-agent` is not found after `pip install -e .`, add the
  Homebrew/Python scripts directory to `PATH` (usually
  `$(python3 -m site --user-base)/bin`).
- Grant Terminal (or your emulator) Full Disk Access if you work with files
  outside your home directory; otherwise file tools may be denied by TCC.

### Termux (Android)
- Install the base packages first:
  ```bash
  pkg update && pkg upgrade
  pkg install python clang make
  ```
- Some native dependencies need a compiler; `clang` covers most wheels.
- Storage access: run `termux-setup-storage` and keep projects under
  `~/storage/shared` or the Termux home, not on removable media.
- Bubblewrap is not available on Android; code-task sandboxing falls back to
  the no-sandbox path, so only run trusted code.
- Memory is tight on phones: keep `ulimit -v 2400000` in mind and avoid
  large RAG indexes on device.
```