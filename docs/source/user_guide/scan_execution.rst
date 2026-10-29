Scan Execution
==============

``instrument.scan(plan, *, on_error="raise")`` executes a verified native scan
and returns a :class:`~mcstasscript.ScanResult`. Configure the instrument with
``instrument.settings(...)`` before calling ``scan``; execution has no
call-level settings override. This is the Phase 7 execution API. Planning and
preflight remain non-executing APIs; execution is deliberately a separate step.

.. code-block:: python

   instrument.settings(ncount=20_000, seed=314159)
   result = instrument.scan(plan)

The execution path has no additional Python dependencies. It is verified only
for McStas ``3.10.0``. McStas ``3.7.14`` is rejected for execution, despite its
separate preflight capability record, because its sequential seed behavior is
not suitable for this execution contract. McXtrace, including unknown and
3.7.x versions, has no verified execution record and is rejected rather than
supported by inference. Other McStas versions are also rejected when their
cached version cannot be verified.

Execution Contract
------------------

Before creating output, ``scan`` runs ``scan_preflight`` and rejects invalid
plans, unsupported settings, an existing output path, partial instrument runs,
and an unverified native tool. The supplied instrument is not used as the
working instrument: execution makes a deep copy, writes and runs that copy,
and never calls ``backengine``. Parameter values, components, run settings,
and output settings on the original instrument are therefore unchanged. All
settings are retained precisely because the original is left untouched rather
than temporarily modified. The isolated fresh-source report internally forces
``force_compile=True``; this does not toggle the original instrument. An
original value of either ``True`` or ``False`` remains unchanged, including
when execution raises ``KeyboardInterrupt``.

The copied instrument is written below a new output root. The source snapshot
is nonempty even when a string parameter needs a native placeholder while the
copy is written. That placeholder is confined to the deep copy; command-line
values provide the runtime values and the original instrument's parameter
values and source state are preserved. A successful or partial run has this
layout:

.. code-block:: text

   scan_root/
   |-- source/
   |   `-- instrument_name.instr
   |-- native/
   |   `-- ... native McStas output ...
   |-- scan_manifest.json
   `-- execution.log

The root must be new. An existing path is an error; the execution API neither
overwrites it nor increments it to choose another folder. The source is
compiled in isolation with the native ``--force-compile`` option, and the
requested native format is selected by the effective ``NeXus`` setting. A
previous compiled instrument is not reused.

Settings And Seeds
------------------

The effective settings start with the instrument's cached run settings and
preflight defaults. Set execution settings with ``instrument.settings(...)``
before calling ``scan``; ``scan`` has no call-level settings override. The
separate ``scan_preflight`` method retains its own ``settings`` argument for
execution-free validation. ``ncount`` is the number of simulated particles
**per scan point**, not the total over the scan. The usual native settings such
as ``mpi``, ``NeXus``, ``openacc``, and ``gravity`` are passed to the native
runner after validation.

``seed=None`` is replaced with a newly generated positive base seed before the
manifest and command are written. The generated value is bounded so that the
largest native offset remains in signed 64-bit range. For a plan with ``N``
points, an explicit base seed ``B`` must satisfy
``B + 1024 * (N - 1) <= 2**63 - 1``. With an explicit seed, the same positive
base seed is recorded. McStas 3.10 applies its point offset of ``1024`` to the
native point index. This records the native seed policy; it is not a promise
that points are statistically independent.

``run_path`` is inherited from the instrument's cached run settings (or its
original input path) and is used as the subprocess working directory, so
resources can be found in the original execution context. It is not changed by
execution; configure persistent instrument settings before ``scan`` when
needed. The copied ``.instr`` source can still contain local component or
include references. Those references must be resolvable from the inherited run
context; inspect such paths before relying on a scan. There is no relocation or
resource-rewriting strategy in this API.

Ordering And Point Identity
---------------------------

The plan's canonical IDs are retained in the manifest-backed result. Grid
plans use the plan order, where the last plan axis changes fastest. Native
McStas ``-M`` list grids use the first command-line axis fastest instead. The
execution mapping compensates for this difference, so native output is mapped
back to canonical plan IDs rather than exposing native storage order.

For example, a plan with axes ``x=[0, 1]`` and ``y=[10, 20, 30]`` has canonical
plan order ``(0, 10), (0, 20), (0, 30), (1, 10), ...``. Native grid order is
``(0, 10), (1, 10), (0, 20), (1, 20), ...``. A grid point's native index and
canonical plan ID are mapped in both directions during loading. Paired plans
and explicit point plans already have the same index order in both systems.
Repeated values remain separate points.

Point metadata keeps native identity separate from native storage identity.
``native_point_id`` is the native scan index, including an index derived from
the verified seed offset when that is the available identity evidence.
``storage_point_id`` and ``storage_entry`` identify the point's position in the
native storage output. They can differ when native entries are omitted or
stored in a different order and must not be treated as interchangeable.

``ScanPlan`` deliberately accepts arbitrary Python string values. That plan
contract is broader than native command representability: preflight checks the
parameter type and instrument constraints, but does not by itself promise that
every string can be encoded by the native runner. The native command builder
performs the additional representability validation after preflight and before
creating the output root or writing the copied source. A failure there writes
no scan output.

Native list values use McStas 3.10's comma-separated list syntax. In a
multi-point scan, a string comma is emitted as ``\,``. Internal spaces,
apostrophes, quotes, and internal backslashes are supported as literal argument
characters because the runner is called without a shell. A trailing backslash
is not supported: the native list parser treats it as an incomplete delimiter
escape. Empty strings and strings containing ``:`` are rejected in list mode.
The same empty-string rejection applies to a single-point scalar value. All C0
control characters (U+0000 through U+001F), DEL (U+007F), and Unicode linebreak
controls such as U+0085, U+2028, and U+2029 are rejected for native execution.
Leading or trailing whitespace is also rejected because scalar native metadata
readers strip it; internal whitespace remains supported.

Files, Manifests, And Results
-----------------------------

``scan_manifest.json`` is the durable link between the requested plan and the
native observations. Its point records are sparse: finalization records only
observed native points with a source or summaries, while absent requested
points have no fabricated record. ``execution["recorded"]`` is therefore the
number of observed/recorded points, not the number requested by the plan;
``execution["point_count"]`` is the requested count. ``ms.load_scan(scan_root)`` or
``ms.load_scan(scan_root / "scan_manifest.json")`` detects it, reloads the
manifest, reads ``scan_root/native``, and retains every point from the plan,
including points for which no native output exists:

.. code-block:: python

   result = instrument.scan(plan)
   reloaded = ms.load_scan(scan_root)
   assert reloaded.plan == plan
   assert [point.point_id for point in reloaded.points] == list(range(plan.point_count))

The process status in ``result.execution`` and the per-point ``status`` are
different facts. Native files do not independently prove that a point ran, so
point status remains ``"unknown"`` unless independent manifest evidence
records a point status. ``point.data_status`` separately reports whether full
data is available, partial, missing, or unreadable. Readable data is not
execution-success evidence.

This distinction also applies when output is incomplete. A partial run, an
all-failed run, or an invalid root HDF5 file still returns all known points
from the plan; points without safely associated native observations remain in
the result as missing data and have no fabricated manifest record. Diagnostics
and process evidence identify why native observations could not be associated
or read.

In particular, a first failed native point can leave the combined NeXus file
invalid or corrupt. Execution does not rerun points, replace the raw output,
repair the HDF5 file, or rebuild a combined file automatically. The raw native
file is preserved. When a partial native text summary is readable, a row with a
unique numeric coordinate match is retained as a summary-only observation;
repeated coordinates are not guessed and remain unassociated with diagnostics.
Missing planned IDs remain the plan's points with no fabricated execution
status or seed observation. Intended seeds can still be derived from the
manifest's base seed and native index, but a missing point's actual seed stays
unknown. This is conservative evidence: a readable summary does not prove a
successful point, and the default error policy raises ``ScanExecutionError``
for the output problem. Use ``on_error="continue"`` when retaining that
partial result is intentional.

Passing ``scan_root / "native"`` explicitly to ``ms.load_scan`` bypasses the
manifest. That is a direct native load: it has no requested plan or manifest
process evidence, and its point statuses remain unknown under the native-load
rules described in :doc:`scan_results`.

Errors And Interruption
-----------------------

The default ``on_error="raise"`` raises ``ms.ScanExecutionError`` after a
native process completes with a nonzero exit or output problems such as missing
points, unreadable data, or failed finalization. Setup, source-writing, and
launch failures use the same exception once a checkpoint exists. The
exception's ``.result`` contains the partial ``ScanResult`` and the checkpoint
remains on disk.

With ``on_error="continue"``, those execution or output problems return the
partial result instead of raising. This does not turn missing data into a
successful point.

If the caller presses Ctrl-C, the native process is stopped, the interrupted
manifest and a checkpoint are written, and ``KeyboardInterrupt`` is re-raised.
The caught exception has a ``.result`` attribute containing the partial result.

There is no live plotting, resume operation, or Python worker orchestration in
this API. Use the returned result or reload the checkpoint after execution.

Concrete Example
----------------

This is a small real instrument. It uses an ASCII native output so the example
does not require a separate data-format choice:

.. code-block:: python

   from pathlib import Path
   from tempfile import TemporaryDirectory

   import mcstasscript as ms

   with TemporaryDirectory() as temporary:
       root = Path(temporary)
       input_path = root / "input"
       input_path.mkdir()

       instrument = ms.McStas_instr(
           "phase7_scan", input_path=str(input_path),
           output_path=str(root / "scan"),
       )
       wavelength = instrument.add_parameter("wavelength", value=3.0)
       source = instrument.add_component("source", "Source_simple")
       source.set_parameters(
           radius=0.001, dist=1.0, focus_xw=0.01, focus_yh=0.01,
           lambda0=wavelength, dlambda=0.1, flux=1e7,
       )
       monitor = instrument.add_component("spectrum", "L_monitor")
       monitor.set_parameters(
           xwidth=0.02, yheight=0.02, nL=24, Lmin=1.0, Lmax=7.0,
           filename='"spectrum.dat"',
       )
       monitor.set_AT([0, 0, 1], RELATIVE=source)

        plan = ms.ScanPlan.points([
            {"wavelength": 2.0},
            {"wavelength": 3.0},
        ])
        instrument.settings(ncount=1_000, seed=314159, NeXus=False)
        result = instrument.scan(plan)

       # Reconcile the durable result before selecting a new output root for
       # any later run.
       loaded = ms.load_scan(root / "scan")
       print(loaded.execution["process"]["status"])
       print(loaded.dataset("spectrum").total_I)

The example requires a local McStas 3.10.0 installation and its normal
component library. The existing native-scan and scan-analysis tutorials remain
useful for manual ``mcrun`` scans and result analysis; this page covers the
Phase 7 managed execution path and does not require a new tutorial notebook.
