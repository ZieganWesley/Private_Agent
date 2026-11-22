## Part H: Platform Support (Linux, WSL, macOS and Termux; no native Windows)

Goal: support **POSIX systems only**, namely Linux (including WSL2), macOS and,
as a partial environment, Termux on Android. Drop native Windows, so it runs
in WSL without any setup trouble.

### H1. What was checked

- **Native-Windows code is almost nonexistent.** The source has no `winreg`,
  `msvcrt`, `os.name == "nt"` or `WindowsPath` code. The Windows traces are:
  - tests that simulate Windows or macOS (`tests/unit/test_providers.py:282`
    patches `sys.platform = "win32"`, `tests/unit/test_code_tasks.py:297–327`,
    `tests/integration/test_agent_runtime.py:210`);
  - README wording about "supported desktop platforms" (lines 32, 139, 350);
  - `isolation_unavailable_reason()` (`code_tasks/__init__.py:77`), which
    already rejects everything except Linux.

  So removing Windows is mostly removing tests and wording, plus a clear
  startup message.
- **The code already relies on POSIX features:**
  - the secure write path uses `O_NOFOLLOW`, `O_DIRECTORY` and `dir_fd`
    (`tools/filesystem.py:216–231`);
  - hard links via `os.link` (`tools/filesystem.py:416`,
    `database/memory.py:2890` and `:2955`);
  - Bubblewrap and `prlimit` for the sandbox.
- **On this machine (WSL2, Fedora 44):**
  - mirrored networking with `hostAddressLoopback=true` is set in
    `.wslconfig`, so `127.0.0.1:11434` can reach a Windows-side Ollama;
  - Ollama is installed on the Windows side but was not answering
    (connection refused), so it was probably not running;
  - unprivileged user namespaces work (`unshare --user` succeeded), and
    `prlimit` is installed, but **`bwrap` is not installed**. That one missing
    package is why shell commands, project tests and stdio MCP are disabled,
    and why 1 of the failing tests fails.

### H2. Target support levels

| Platform | Level | Sandbox | Local models | RAG | Media |
|---|---|---|---|---|---|
| Linux (x86_64, arm64) | Full | Bubblewrap + `prlimit` | Ollama, LM Studio, llama.cpp | Full | Full |
| WSL2 | Full | Bubblewrap + `prlimit` inside WSL | Ollama in WSL, or Windows-side Ollama via mirrored networking | Full | Limited: no webcam or microphone by default |
| macOS (arm64, x86_64) | Full | Seatbelt (`sandbox-exec`) + `setrlimit` (step 6.14) | Ollama, LM Studio, llama.cpp | Full | Full (needs camera and microphone permission) |
| Termux (Android) | Partial | None available; commands off by default | Ollama or llama.cpp running in Termux, or an online provider | BM25 keyword search only, unless the vector store installs | Off |
| Native Windows | Not supported | — | — | — | — |

On native Windows, startup prints a short message pointing to WSL and exits.

### H3. Issues to fix per platform

#### All platforms

**H3a. Hard dependency on heavy packages.**
`runtime.py` imports `ollama`, `langchain_ollama`, the RAG stack (Chroma) and
the media module at the top (`runtime.py:24`, `:32`, `:120`, `:189`). If any
of them fails to install, as Chroma's `onnxruntime` dependency is likely to on
Termux, the whole app fails to start. Optional parts should be imported only
when used, with a clear message when missing.

**H3b. No platform detection or doctor command.**
Nothing reports which features are available on the current system or why
some are missing. The Ready summary shows the OS and whether isolation works,
but not how to fix it.

**H3c. Platform claims are not tested.**
There is no CI matrix, so macOS and arm64 support are untested.

#### WSL

**H3d. Missing Bubblewrap with no hint.**
The message says to install Bubblewrap and `prlimit` but not the command for
the current distribution (`sudo dnf install bubblewrap` on Fedora,
`sudo apt install bubblewrap` on Ubuntu or Debian).

**H3e. Reaching Ollama on the Windows side.**
The default `http://localhost:11434` works only with mirrored networking (as
on this machine) or with Ollama installed inside WSL. With NAT networking (the
WSL default), Windows-side Ollama is at the host's gateway IP and needs
`OLLAMA_HOST=0.0.0.0` on Windows. The loopback-only check in
`agent/local_runtimes.py:28` would also reject the gateway address. None of
this is documented.

**H3f. Working on `/mnt/c` is slow and unreliable.**
Projects on the Windows drive (`/mnt/c/...`) go through a translation layer.
File access is much slower, hard links and `O_NOFOLLOW` behave differently,
and file permissions do not map cleanly. This can break the secure write path
and snapshots. Nothing warns about it.

**H3g. No webcam or microphone by default.**
WSL has no direct access to cameras, and microphone access depends on WSLg's
audio support. The media tools should report this clearly rather than fail.

#### macOS

**H3h. No sandbox.**
Shell, project tests and stdio MCP are disabled because Bubblewrap is
Linux-only (C4). This needs a Seatbelt backend and `resource.setrlimit`
instead of `prlimit` (step 6.14).

**H3i. Camera and microphone permissions.**
The terminal app must be granted access in System Settings, and failures
should say so.

#### Termux (partial environment)

**H3j. No sandbox available.**
Android does not allow unprivileged user namespaces, so Bubblewrap cannot
run, and `proot` is not a security boundary. Under the current fail-closed
rule, commands stay disabled. The plan offers an explicit, off-by-default
choice (see H4); whether to allow it is a decision for the project owner.

**H3k. Packages that may not install.**
Chroma needs `onnxruntime`, which has no Android builds, and some packages
need to be compiled in Termux. RAG needs a fallback that works without them.

**H3l. Shared storage limits.**
`/sdcard` (shared Android storage) does not support hard links or symlinks
the same way, which can break snapshots and the secure write path. The
workspace should live in Termux's home folder.

**H3m. Small screens and limited input.**
Wide tables, long prompts and Ctrl or Esc shortcuts are hard to use on a
phone keyboard. The UI needs a compact layout and keys that exist on the
Termux extra-keys row.

### H4. Approach

1. **POSIX only.** Declare it in `pyproject.toml` (`Operating System ::
   POSIX` classifiers), and add a startup check that exits on native Windows
   with a message pointing to WSL. Remove the tests that simulate Windows.
2. **A platform module.** Add `platform_support.py`, which detects Linux,
   WSL (from `/proc/sys/fs/binfmt_misc/WSLInterop` or a `microsoft` kernel
   name), macOS and Termux (`PREFIX` containing `com.termux`). It reports which
   features are available and why. Everything else asks this module instead
   of checking `platform.system()` itself.
3. **A sandbox interface with backends.** Bubblewrap (Linux and WSL),
   Seatbelt (macOS) and None. With the None backend, commands stay disabled
   unless the user turns on a clearly labelled "unsandboxed, approve every
   command" option for that session. That option must never apply in Full
   mode without per-command approval.
4. **Optional extras.** Split dependencies into a small core plus extras:
   `rag` (Chroma and embeddings), `media`, and `all`. The core must install
   on Termux. Without the `rag` extra, document search falls back to BM25
   only (`rank-bm25` is pure Python) or to SQLite FTS5, which is already used
   for memory.
5. **A `private-agent doctor` command.** It lists every feature as available
   or missing, with the reason and the exact fix command for the detected
   system (for example `sudo dnf install bubblewrap`, `pkg install ollama`,
   or "move the project out of `/mnt/c`").
6. **Setup docs per platform:** WSL (Bubblewrap, Ollama inside WSL vs.
   Windows-side with mirrored networking, keeping projects on the Linux
   filesystem), macOS (Homebrew, permissions) and Termux (packages, storage
   location, limits).
7. **A CI matrix** on Ubuntu x86_64, Ubuntu arm64 and macOS, plus a
   core-only install test that mimics Termux (no `rag` or `media` extras).

---

