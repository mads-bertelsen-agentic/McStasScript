# McStas 3.10.0 Native Scan Fixtures

This is a separate, versioned compatibility set. It does not replace or
rewrite the original McStas 3.7.14 fixtures in the parent directory. The five
case directories contain only native `.sim`, `.dat`, and `.h5` files copied
byte-for-byte from native `mcrun` output. The copied instrument, generated C,
binary, logs, and temporary build directory are not shipped.

## Provenance

The captures were made on 2026-10-28 in the `mcstasscript_mcstas` environment:

| Item | Value |
| --- | --- |
| `mcrun` | 3.10.0 |
| `mcstas --version` | mcstas code generator version 3.10.0 (git) |
| Python | 3.14.6, conda-forge, Clang 20.1.8 |
| NumPy / h5py | 2.5.1 / 3.16.0 |
| HDF5 / NeXus root | 1.14.6 / 4.4.2 |
| Instrument SHA-256 | `18cad4159348e443ee9aafd292d7198b5a997ce2108e45fd463def2abd034447` |
| Base seed / count | `314159` / `64` per point |

The full commands, environment, per-file byte sizes and SHA-256 hashes, point
metadata, entry metadata, and literal summary rows are in `manifest.json`.
`verify.py` reads only these native files and uses no package scan loader or
subprocess.

## Cases And Changes

| Case | Native observation |
| --- | --- |
| `ascii_hashseed_1` | Header variables are `wavelength theta ...`, while rows are `theta wavelength ...`; `PYTHONHASHSEED=2` produced the opposite header order in the non-shipped comparison probe. |
| `ascii_categorical_repeated` | Category values are represented by per-list numeric indices in the summary; labels remain in point `.sim` metadata. Points 0 and 2 have identical coordinates but remain separate and have seeds `314159` and `316207`. |
| `ascii_partial` | Four points are declared; point 1 retains `mccode.sim` metadata but no detector file, while points 2 and 3 continue and complete. The scanner returns zero even though the child failure status is 23. |
| `nexus_partial` | One combined HDF5 file has `entry1`, `entry2`, and `entry3` for points 0, 2, and 3. Failed point 1 has no entry. The summary declares four points but contains three rows. |
| `nexus_parallel` | `--scan_split=2` produces one merged root `mccode.h5` with `entry1` and `entry2`, not separate retained per-point HDF5 files. |

Relative to the 3.7.14 baseline, 3.10.0 shifts both sequential and parallel
point seeds by `1024 * point_index`, supports unequal Cartesian counts through
comma-separated `-N` and list products through `-L -M`, and fixes the
`--scan_split=2 -L` summary failure. The single `-M -N N` form still declares
the per-axis N rather than the total row count. Root ASCII `mccode.sim` still
has the missing `yvars` closing parenthesis and missing separator before the
first detector variable.

Native NeXus entry names are one-based storage names, not original external
point IDs. The entries contain parameter metadata and seeds, but no explicit
zero-based plan-ID dataset. In the partial NeXus case, entry numbers happen to
remain `entry1`, `entry2`, `entry3` for successful points 0, 2, and 3; that is
not sufficient evidence to identify an omitted point in general. Do not infer
unattempted points or deduplicate repeated coordinates from entry names.

Changing `--format=McCode` to `--format=NeXus` without `--force-compile` does
not rebuild an existing binary. The ASCII-only binary warns that NeXus is
unsupported and falls back to McCode output. Genuine NeXus captures therefore
use `--force-compile`, as recorded in this manifest.

## Source References

The installed 3.10.0 scanner source inspected for these observations is:

- `/Users/user/miniforge3/envs/mcstasscript_mcstas/share/mcstas/tools/Python/mcrun/optimisation.py:51-126` for header set order and declared `Numpoints`.
- `/Users/user/miniforge3/envs/mcstasscript_mcstas/share/mcstas/tools/Python/mcrun/optimisation.py:282-323` for per-parameter categorical index resolution.
- `/Users/user/miniforge3/envs/mcstasscript_mcstas/share/mcstas/tools/Python/mcrun/optimisation.py:343-389` for Cartesian recursion and order.
- `/Users/user/miniforge3/envs/mcstasscript_mcstas/share/mcstas/tools/Python/mcrun/optimisation.py:395-411` for point seeds.
- `/Users/user/miniforge3/envs/mcstasscript_mcstas/share/mcstas/tools/Python/mcrun/optimisation.py:487-641` for sequential failure handling, row order, and NeXus append.
- `/Users/user/miniforge3/envs/mcstasscript_mcstas/share/mcstas/tools/Python/mcrun/optimisation.py:644-778` for split NeXus merging.
- `/Users/user/miniforge3/envs/mcstasscript_mcstas/share/mcstas/tools/Python/mcrun/mcrun.py:806-844` for list/product modes and total counts.
- `/Users/user/miniforge3/envs/mcstasscript_mcstas/share/mcstas/tools/Python/mcrun/mccode.py:147-216` for timestamp-based binary reuse.
- `/Users/user/miniforge3/envs/mcstasscript_mcstas/share/mcstas/tools/Python/mcrun/mccode.py:402-497` for runtime format forwarding.

## Reproduction

From a fresh temporary directory containing a copy of the parent fixture's
`tiny_scan.instr`, activate the environment and run these exact commands:

```bash
source "$HOME/miniforge3/etc/profile.d/mamba.sh" && mamba activate mcstasscript_mcstas
PYTHONHASHSEED=1 mcrun --force-compile --format=McCode --seed=314159 --ncount=64 --dir=ascii_hashseed_1 tiny_scan.instr -N 3 theta=-1,1 wavelength=2,6
mcrun --force-compile --format=McCode --seed=314159 --ncount=64 --dir=ascii_categorical_repeated tiny_scan.instr -L theta=0,1,0 wavelength=3,5,3 category=vanadium,nickel,vanadium
mcrun --force-compile --format=McCode --seed=314159 --ncount=64 --dir=ascii_partial tiny_scan.instr -L theta=0,1,2,3 fail=0,1,0,0
mcrun --force-compile --format=NeXus --seed=314159 --ncount=64 --dir=nexus_partial tiny_scan.instr -L theta=0,1,2,3 fail=0,1,0,0
mcrun --force-compile --format=NeXus --seed=314159 --ncount=64 --dir=nexus_parallel tiny_scan.instr --scan_split=2 -N 2 theta=-1,1
```

Do not copy generated instruments, C, binaries, or logs into this directory.
The broader temporary probe script used to select these cases is not shipped.
Run verification from the repository root with:

```bash
python mcstasscript/tests/scan_fixtures/mcstas_3_10/verify.py
```
