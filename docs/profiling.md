# Profiling and Performance Notes

## Representative workload
Profiled `RepositoryIndex.index_repository()` over this repository (185 supported
source files, 2,600 symbols) with `cProfile`.

## Hotspots
| Function | Time (s) | Type |
|---|---:|---|
| `sqlite3.Connection.commit` | 0.674 | I/O-bound / storage |
| `RepositoryIndex._walk` | 0.439 | CPU-bound parsing |
| `tree_sitter.Parser.parse` | 0.269 | CPU-bound parsing |
| `sqlite3.Cursor.execute` | 0.240 | I/O-bound / storage |

The tree-sitter parse and AST walk are independent per file and CPU-bound.

## Change
`index_repository()` now collects candidate files, performs database bookkeeping,
then parses all candidate files through `_parse_files_parallel()`. For 8 or more
files on a multi-core machine it uses `ProcessPoolExecutor` with a chunk size of
`max(1, file_count // (cpu_count * 4))`. Smaller batches and single-core or
unavailable multiprocessing fall back to in-process parsing. Results are applied
in input order, preserving prior behavior.

## Before / after
- Before: 1.847 s
- After: 1.145 s
- Improvement: ~0.702 s (about 38%)

The remaining time is dominated by SQLite commits and file I/O.
