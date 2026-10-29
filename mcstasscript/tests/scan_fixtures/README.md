# Native Scan Fixtures

Small, genuine McStas outputs for scan-loader and scan-model tests. These are
fixtures only, not an execution API. `capture.py` never imports McStasScript.
All files below the twelve output directories were produced by native `mcrun`
and copied byte-for-byte, including incomplete and empty native output files.
Only automatic `.instr`/`.c` copies and build products are excluded. Do not fix
headers, synthesize missing rows, recompress HDF5, or deduplicate repeated points.

## Provenance

Captured on 2026-10-28 using the installed conda-forge
`mcstasscript_mcstas` environment on macOS arm64:

| Tool | Observed Version |
| --- | --- |
| `mcrun --version` | `3.7.14` |
| `mcstas --version` | `mcstas code generator version 3.7.14 (git)` |
| Native NeXus writer, HDF5 root `NeXus_version` | `4.4.2` |
| HDF5, native file root and h5py | `1.14.6` |
| Python | `3.14.6`, conda-forge, Clang `20.1.8` |
| h5py | `3.16.0` |
| NumPy | `2.5.1` |

The initial inspection used `mcrun --help`, `mcrun --showcfg=tooldir`,
`mcrun --showcfg=resourcedir`, and the installed sources
`tools/Python/mcrun/{mcrun.py,optimisation.py,mccode.py}`. In particular,
`LinearInterval`, `MultiInterval`, `Scanner`, `Scanner_split`,
`_simulate_point`, and both summary-header writers explain the layouts below.
No native source or installed configuration was modified.

`tiny_scan.instr` uses `Source_simple` and one four-bin `L_monitor` named
`spectrum`. Parameters are numeric `theta` and `wavelength`, string `category`,
and integer `fail`. Category and theta change the source intensity; wavelength
changes the occupied monitor bin. Every successful point launches and detects
exactly **64** neutrons. The base seed is **314159**.

`expected.json` separates independently specified `intended_points` from
`observed` native metadata. It includes every point's intended parameters, seed,
and status; observed ordering, incomplete points, missing detector-point IDs,
literal summary rows and column labels; exact native command argument vectors;
exit codes; source SHA-256; and byte sizes/SHA-256 for every retained artifact.
The recorded build directory is provenance, not a runtime dependency.

## Capability Matrix

| Case | ASCII Directory | NeXus Directory | Observed Points and Order |
| --- | --- | --- | --- |
| 1D range, `-N 3` | `ascii_1d` | `nexus_1d` | theta `-1, 0, 1`, wavelength fixed at `3` |
| Unequal Cartesian 2x3, explicit `-L` | `ascii_cartesian_2x3` | `nexus_cartesian_2x3` | `(-1,2), (1,2), (-1,4), (1,4), (-1,6), (1,6)`; theta fastest |
| Paired numeric ranges, `-N 3` | `ascii_paired` | `nexus_paired` | `(-1,2), (0,4), (1,6)`; three points, not nine |
| Paired lists, categories, exact repeated coordinates | `ascii_categorical_repeated` | `nexus_paired_categorical_repeated` | `(0,3,vanadium), (1,5,nickel), (0,3,vanadium)`; points 0 and 2 remain distinct |
| Parallel `--scan_split=2` | `ascii_parallel` | `nexus_parallel` | ASCII: native `-M -N 2`, four points `(-1,3), (1,3), (-1,5), (1,5)`; NeXus: two theta points `-1,1` at wavelength `3` |
| Intentional runtime failure at point 1 | `ascii_partial` | `nexus_partial` | Four intended points; point 0 complete, point 1 metadata only, points 2 and 3 unattempted |

Sequential ASCII writes root `mccode.sim` and `mccode.dat`, plus zero-based
`0/`, `1/`, etc. Each completed point has `mccode.sim` and `spectrum.dat`.
The detector is a four-row, four-column ASCII array: `L I I_err N`.
Its wavelength centers are `1.75, 3.25, 4.75, 6.25` Angstrom.

Sequential NeXus writes **one combined `mccode.h5`**, with `entry1`, `entry2`,
etc. in scan order. Each entry's parameters are string-valued datasets under
`simulation/Param`, also present as attributes on that group. Seed is an
attribute of `simulation`. Detector data live under `data/spectrum_dat`, with
four-element datasets `Wavelength__AA_`, `data`, `errors`, and `ncount`.
The native component-output hierarchy links to the same detector objects.
The scan's `mccode.dat` is genuinely zero bytes; there is **no root
`mccode.sim`**. Do not treat that empty summary as an empty scan.

Parallel ASCII also writes numbered point directories. Workers may finish in
any order, but root summary rows are sorted back into point-index order.
Point seeds are `314159 + 1024 * point_index`, observed as
`314159, 315183, 316207, 317231` in the four-point capture. Sequential runs
reuse `314159` at every point; repeated identical points therefore have
identical detector arrays. Parallel NeXus is **not combined**: it writes
`0/mccode.h5`, `1/mccode.h5`, each with its own `entry1`. Its root summary is
also empty and root `mccode.sim` is absent.

## Native Limitations

- `-M -N N` uses the same N on every axis. Native recursion makes the first
  supplied parameter fastest. A native unequal 2x3 range scan is not expressible
  through `-M`; these fixtures explicitly enumerate exactly six points with
  `-L`. The artifacts alone cannot distinguish that Cartesian intent from an
  arbitrary paired list. The sidecar records the independent intended points.
- `-N` without `-M` interpolates every supplied range together, not a product.
  `-L` also zips lists. Lists must have equal lengths.
- The ASCII `mccode.dat` header builds parameter names from a Python set,
  whereas data rows follow command-line parameter order. Several captured
  headers say `wavelength theta`, but their rows are **theta then wavelength**.
  Header order may change with Python hash randomization. Resolve coordinates
  against point metadata, not just that header. Native headers are unmodified.
- Native `-M` summaries claim `Numpoints: 2` / `multiarray_1d(2)` for the
  four-point 2x2 capture. The count is per axis, not total rows.
- Root ASCII `mccode.sim` also has native formatting defects: a missing space
  between the last parameter and first intensity variable, and missing closing
  parentheses in `yvars`. Point `mccode.sim` remains authoritative for parameters.
- For a sequential `-L` row containing any nonnumeric parameter, the entire
  coordinate prefix is replaced with the zero-based row index. Thus the
  categorical summary has only three columns even though its header declares
  five. Neither category nor the other coordinates can be reconstructed from
  that summary alone; all survive in point metadata and NeXus entries.
- Both partial captures use the same deliberate native `exit(23)` in
  `INITIALIZE`. `mcrun` exits **1**, reporting the child exit status **23**.
  ASCII retains `1/mccode.sim` but no `1/spectrum.dat`, and only point 0 has a
  summary row despite the header declaring four points. NeXus retains readable
  `entry2` parameter metadata, but its `data` group is empty and `end_time` is
  absent. `entry1` is complete. Later points have no files or entries. Native
  artifacts do not contain an explicit failed/unattempted status table or a
  complete intended plan; those distinctions are independent sidecar knowledge.
- Fixture IDs are zero-based external plan IDs. Sequential NeXus provides
  one-based entry names, not an explicit scan point-ID dataset. Do not deduplicate
  entries by coordinates or infer future/unattempted points from existing entries.
- Native ASCII `--scan_split=2 -L` has a separate summary-writer failure in
  this version. A direct probe (not retained as another fixture) ran both points
  successfully, then exited 1 with
  `TypeError: %i format: a real number is required, not NoneType`, because
  `options.numpoints` is unset. Its exact command in the recorded capture build
  directory was `mcrun --format=McCode --seed=314159 --ncount=64 --dir=probe_split_list tiny_scan.instr --scan_split=2 -L theta=0,1`.
  The parallel fixtures therefore use supported `-N`/`-M` paths, not `-L`.
- Switching `--format=NeXus` after compiling an ASCII-only binary does not
  automatically force a rebuild. A probe reused that binary, warned that NeXus
  was unsupported, fell back to ASCII, and later failed reading the appended
  summary. Every retained capture uses `--force-compile`, so all `.h5` files
  are genuine native NeXus, not converted ASCII. An early instrument expression
  containing a quoted string also exposed an unescaped NeXus-generated C string;
  the final instrument computes `source_flux` in `INITIALIZE` instead.

## Reproduce and Verify

From the repository root:

```bash
source "$HOME/miniforge3/etc/profile.d/mamba.sh" && mamba activate mcstasscript_mcstas
python mcstasscript/tests/scan_fixtures/capture.py --replace
python mcstasscript/tests/scan_fixtures/capture.py --verify
```

Without `--replace`, capture refuses to overwrite existing fixture directories.
Each invocation creates a new `scan-fixtures-*` build directory under the
portable system temporary directory returned by `tempfile.gettempdir()`.
Optionally pass `--work-root PATH` to select an existing approved temporary
parent. For native captures in the original local workspace, use:

```bash
python mcstasscript/tests/scan_fixtures/capture.py --replace --work-root /var/folders/1t/8gxgl7l51dd3pk_tbw0mvtj80000gn/T/opencode
```

Native binaries, generated C, debug products, and full capture logs stay there,
never in repository artifact directories. The script invokes subprocesses with
argument lists and no shell. `--replace` replaces only its named generated
fixture directories and its generated `expected.json`, not instrument/script/docs.
Dates and native header hash ordering make whole-file regeneration hashes
time-dependent; the deterministic seeds reproduce detector arrays.

For each retained capture the exact command is the `command` array in
`expected.json`, executed in the recorded temporary build directory containing
the single copied `tiny_scan.instr`. There are no hidden postprocessing commands.
For example, the unequal grid commands are:

```bash
mcrun --force-compile --format=McCode --seed=314159 --ncount=64 --dir=ascii_cartesian_2x3 tiny_scan.instr -L theta=-1,1,-1,1,-1,1 wavelength=2,2,4,4,6,6
mcrun --force-compile --format=NeXus --seed=314159 --ncount=64 --dir=nexus_cartesian_2x3 tiny_scan.instr -L theta=-1,1,-1,1,-1,1 wavelength=2,2,4,4,6,6
```

`--verify` needs only NumPy and h5py, not McStas. It checks every retained file's
SHA-256, source provenance, planned versus observed point parameters/seeds/order,
failed-point metadata, summary coordinates and row counts, detector shapes,
finite values, and 64 detected events per completed point. It also compares the
five matching sequential ASCII/NeXus cases, allowing ASCII's six-significant-digit
rounding (`rtol=1e-5`, no absolute tolerance), and checks exact NeXus array identity
at repeated coordinates. This verification is independent of package loaders.
The base tests in `mcstasscript/tests/test_scan_fixtures.py` run it with
subprocess execution blocked and check the categorical-summary and incomplete
point metadata. Run them without a native toolchain using
`python -m unittest mcstasscript.tests.test_scan_fixtures -v`.

Native artifacts total **1,076,040 bytes**, including **1,029,096 bytes** across
seven HDF5 files; the largest is the six-entry grid at **336,888 bytes**. The
remaining files are small native summaries, detector arrays, and point metadata.
This permits keeping both formats for every representative case without large
arrays, Cartesian expansion, binaries, or generated C in the repository.

Packaging must graft this entire directory into the source distribution,
including the `.h5`, `.dat`, `.sim`, `.instr`, and `.json` files, using
`graft mcstasscript/tests/scan_fixtures` in `MANIFEST.in`.

For the separately versioned McStas 3.10.0 compatibility probe, see
[`mcstas_3_10/README.md`](mcstas_3_10/README.md). It does not replace these
3.7.14 native fixtures.
