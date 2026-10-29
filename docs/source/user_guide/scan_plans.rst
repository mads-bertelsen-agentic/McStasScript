Scan Plans, Preflight, And Manifests
====================================

This page documents the execution-free scan planning API. A
:class:`~mcstasscript.ScanPlan` describes the requested points without
expanding a Cartesian product. ``scan_preflight`` checks that plan against an
instrument without compiling, running, or changing it. A
``ScanManifest`` records the plan and sparse observations as versioned JSON
metadata. To execute a verified native scan from a plan, use
:meth:`~mcstasscript.McStas_instr.scan`; see :doc:`scan_execution` for its
isolated output and result-loading contract.

Planning and preflight do not write scan output files. ``load_scan`` consumes a
manifest when one is produced by ``instrument.scan``; direct native output
without a manifest continues to use the execution-free native readers.

Creating A Plan
---------------

``ScanPlan`` is available from the package namespace:

.. code-block:: python

   import mcstasscript as ms

   plan = ms.ScanPlan.grid(
       {"theta": [0.0, 1.0]},
       wavelength=[2.0, 2.5, 3.0],
   )

The first positional argument is an optional mapping of parameter names to
axis values. Keyword axes are appended in keyword insertion order. Supplying
the same name in both places is an error rather than an implicit override:

.. code-block:: python

   # Raises ValueError: theta was supplied twice.
   ms.ScanPlan.grid({"theta": [0.0]}, theta=[1.0])

The positional mapping also leaves ``parameters`` available as an ordinary
axis name:

.. code-block:: python

   plan = ms.ScanPlan.grid(
       {"theta": [0.0]},
       parameters=["a,b", '"quoted"', ""],
   )

There are three plan kinds:

* ``ScanPlan.grid(mapping, **axes)`` creates the Cartesian product of its
  axes.
* ``ScanPlan.paired(mapping, **axes)`` takes one value from each axis at the
  same position. All axes must have the same length.
* ``ScanPlan.points(rows)`` takes an explicitly ordered nonempty list or tuple
  of mapping rows. Every row must have the same keys.

Axis and point values are preserved as explicit values. A singleton axis is
still an axis, and repeated values are not deduplicated:

.. code-block:: python

   grid = ms.ScanPlan.grid(angle=[1.0], mode=["low", "low"])
   assert grid.point_count == 2

   paired = ms.ScanPlan.paired(angle=[1.0, 1.0], mode=["low", "high"])
   points = ms.ScanPlan.points([
       {"angle": 1.0, "mode": "low"},
       {"angle": 1.0, "mode": "low"},
   ])

The two rows in ``points`` remain two distinct point IDs even though their
parameter values are equal. ``grid`` likewise has two points because the
explicit ``mode`` axis contains two entries.

Axis values must be an explicit list, tuple, ``range``, or one-dimensional
NumPy array. Empty axes, scalar values, iterators, mappings, sets, and strings
used as an axis are rejected. Individual values may be finite integers, real
numbers, or strings. Boolean values, ``None``, complex values, ``NaN``, and
infinities are rejected. Point rows are mappings, not arbitrary iterables.

Point Order And IDs
-------------------

Grid order is logical and deterministic: the last axis changes fastest. It is
based on the order of the axes in the plan, not on native summary columns,
file output order, or any order chosen by McStas/McXtrace:

.. code-block:: python

   plan = ms.ScanPlan.grid({"x": [1, 2], "mode": ["a", "b", "c"]})

   assert plan[0].point_id == 0
   assert plan[0].parameters == {"x": 1, "mode": "a"}
   assert plan[1].parameters == {"x": 1, "mode": "b"}
   assert plan[3].parameters == {"x": 2, "mode": "a"}
   assert plan[4].grid_indices == {"x": 1, "mode": 1}

Point IDs are zero-based integers. A grid point has ``grid_indices`` mapping
each axis name to its zero-based axis index. Paired and explicit points have
``grid_indices is None``; their integer ``point_id`` is respectively the
paired position or input row position.

The plan stores normalized axes and rows, not the Cartesian product. Axis
inputs are copied during construction, including ``range`` inputs, so
construction costs O(sum of axis lengths); individual ranges are not retained
as lazy iterators. A large grid can nevertheless be created and indexed
without materializing all its points:

.. code-block:: python

   large = ms.ScanPlan.grid(
       a=range(1000), b=range(1000), c=range(1000),
       d=range(1000), e=range(1000),
   )
   assert large.point_count == 1000 ** 5
   point = large[1000]       # materializes this point only

Iteration intentionally visits every point, so use indexing or ``preview``
when inspecting a large plan.

Preview And JSON
----------------

``preview`` returns a detached, bounded summary. It includes the plan kind,
parameter names, total point count, ordering rule, and the first requested
number of materialized points. For axis plans, every axis is summarized by a
mapping with ``length``, bounded ``values``, and a boolean ``truncated`` flag.
The default bound is five; zero returns no point records and no axis values,
but does not expand the grid:

.. code-block:: python

   summary = large.preview(3)
   assert summary["order"] == "last_axis_fastest"
   assert len(summary["points"]) == 3
   assert summary["axes"]["a"] == {
       "length": 1000, "values": [0, 1, 2], "truncated": True,
   }

``to_dict`` returns a detached normalized representation containing either
``axes`` or ``rows``. Unlike ``preview``, ``to_dict`` contains the complete
axis values and is not bounded. It is JSON-safe for all accepted plan values,
and ``from_dict`` strictly restores that representation:

.. code-block:: python

   import json

   document = plan.to_dict()
   encoded = json.dumps(document, allow_nan=False)
   restored = ms.ScanPlan.from_dict(json.loads(encoded))
   assert restored == plan

The serialized axis form contains exactly ``kind``, ``parameter_names``, and
``axes``. The point form contains exactly ``kind``, ``parameter_names``, and
``rows``. Unknown fields or mismatched names are rejected. The plan and its
point views are immutable; changing a returned dictionary or the original
input after construction does not change the plan.

Instrument Preflight
--------------------

An instrument provides a thin method around the helper:

.. code-block:: python

   report = instrument.scan_preflight(
       plan,
       settings={"ncount": 100_000, "seed": 42},
   )

The equivalent helper is ``mcstasscript.helper.scan_preflight.scan_preflight``.
The plan argument must be a ``ScanPlan``; a dictionary is not accepted. The
settings mapping is optional. An empty mapping means that the instrument's
cached run settings and preflight defaults are used.

Preflight is detached and non-mutating. It reads parameter values and cached
instrument metadata, validates copies of parameters, verifies the configured
runner and paths, and returns a report. It does not call
``write_full_instrument``, ``settings`` or ``set_parameters``, compile, launch
a subprocess, execute a runner, create an output directory, or increment an
output-folder name. Runner verification uses ``shutil.which`` within the
configured executable directory and anchors the resolved command to an absolute
path. It does not execute that command.

The report exposes:

* ``valid``: ``True`` only when there are no errors.
* ``errors``: immutable diagnostic messages that make the report invalid.
* ``warnings``: non-fatal diagnostic messages.
* ``plan``: a detached copy of the plan.
* ``fixed_parameters``, ``parameter_snapshot``, ``effective_settings``,
  ``tool``, and ``output_path``: detached preflight context. The mapping
  fields are recursively read-only; nested mappings cannot be changed and
  nested sequences are tuples.
* ``raise_for_errors()``: raises ``ValueError`` with all errors when invalid.
* ``to_dict()``: validates and returns a JSON-safe report document, including
  ``valid``. Unsupported nested values, non-string mapping keys, sets, or
  non-finite numbers raise instead of being converted to fabricated strings.

For example:

.. code-block:: python

   if not report.valid:
       print("Cannot run this scan:")
       for error in report.errors:
           print(" -", error)
       report.raise_for_errors()

Parameter validation checks both the declared or inferred McCode type and the
parameter's own constraints. Supported types are ``double``, ``int``, and
``string``. A double must be a finite real number, an int must be an integral
real number, and a string must be a Python string. Boolean values are not
accepted as numeric values. Constraint checks are performed on deep copies,
so preflight cannot change the instrument parameter.

Every parameter named by the plan must exist. A scanned parameter may have an
unset current value when its explicit scan values establish a supported type;
those scan values are checked. An unset fixed parameter is an error, and a
fixed parameter with a value is checked against its type and constraints. The
report also retains fixed parameters separately from the parameter snapshot.

The selected output path must not already exist. An existing path is an error,
not a request to increment or replace it. A new path is only checked for a
usable existing parent; preflight does not create it. Partial instrument runs
configured through ``run_from`` or ``run_to`` are rejected.

Settings Overrides
~~~~~~~~~~~~~~~~~~

Only these keys are accepted in the preflight ``settings`` mapping:

``ncount``, ``seed``, ``mpi``, ``NeXus``, ``openacc``, ``gravity``,
``force_compile``, ``increment_folder_name``, ``custom_flags``,
``suppress_output``, ``checks``, ``save_comp_pars``, ``executable``,
``executable_path``, ``package_path``, and ``output_path``.

Unknown keys are errors. ``run_path`` and arbitrary runner options are not
override keys. ``ncount`` must be a positive integer;
``seed`` is ``None``, a positive integer, or a decimal integer string; ``mpi``
is ``None``, ``"auto"``, or a positive integer. Boolean options must be
booleans. Nonempty ``custom_flags`` are rejected because they cannot be safely
preflighted. ``executable`` must be the native runner name (``mcrun`` for
McStas or ``mxrun`` for McXtrace). ``executable_path`` must be a nonempty existing
directory, and the runner must be found there. A relative directory is resolved
against the current working directory; an empty, ``PATH``-only setting is not
supported by the scan API.
An ``executable``, ``executable_path``, or ``package_path`` override is
accepted only when it is identical to the corresponding cached instrument
runner value; it cannot reconfigure the runner during preflight.

Overrides are used to build ``effective_settings`` in the detached report and
do not persist to the instrument. In particular, an ``output_path`` override
does not change ``instrument.output_path``. ``package_path`` is retained in
the effective settings as part of the cached runner provenance, but an
override still must match the cached value. The tool's ``path_evidence``
specifically reports executable-path resolution.

Native Capabilities
~~~~~~~~~~~~~~~~~~~

Capability validation uses the package and exact version already cached on the
instrument. It does not invoke ``mcrun``, ``mxrun``, or a version probe. The
tool record exposes ``capability_evidence`` (currently cached instrument
version and verified fixtures) and ``path_evidence``. The latter records the
configured executable path, resolved runner, and ``checked_without_execution``
so path evidence is not confused with an execution result.
Currently verified native capability records are:

* McStas ``3.7.14``: numeric grids, paired scans, and explicit points; no
  verified categorical grids; NeXus support; seed policy
  ``sequential_reuse``.
* McStas ``3.10.0``: numeric and categorical grids, paired scans, and explicit
  points; NeXus support; seed policy ``point_offset_1024``.

McStas ``3.7.14`` adds a warning that sequential points reuse the base seed.
This records a known version policy but does not promise execution behavior.
A categorical grid on that version is an error. Other cached McStas versions
cannot be verified by this API and are errors rather than guessed capabilities.
McXtrace has no verified native capability record here and is reported as
unverified; preflight does not claim support by inference.

For a local check in the ``mcstasscript_mcstas`` environment, construct the
instrument with a real temporary input directory and let the calibrated
defaults select the package and runner. Do not pass ``package_path="/"`` or
``executable_path="/"``: those are real but unusable runner settings. The
constructor may perform its normal setup and version discovery; the assertion
below applies to the later preflight call:

.. code-block:: python

   from pathlib import Path
   from tempfile import TemporaryDirectory
   from unittest.mock import patch
   import json
   import mcstasscript as ms

   with TemporaryDirectory() as directory:
       root = Path(directory)
       instrument = ms.McStas_instr(
           "scan_example",
           input_path=str(root),
           output_path=str(root / "results"),
       )
       instrument.add_parameter("double", "theta", value=0.0)
       instrument.add_parameter("int", "fixed", value=4)
       plan = ms.ScanPlan.points([{"theta": 0.0}, {"theta": 1.0}])

       with patch("subprocess.check_output",
                  side_effect=AssertionError("preflight probed a tool")), \
             patch("subprocess.run",
                   side_effect=AssertionError("preflight executed a run")):
            report = instrument.scan_preflight(plan, settings={"seed": 42})

       assert report.valid
       report.raise_for_errors()

       from mcstasscript.data.scan_manifest import ScanManifest

       manifest = ScanManifest.from_preflight(report)
       restored = ScanManifest.from_dict(json.loads(
           json.dumps(manifest.to_dict(), allow_nan=False)))
       assert restored.plan == report.plan
       assert restored.fixed_parameters == {"fixed": 4}

The example is expected to be valid with the calibrated McStas installation in
``mcstasscript_mcstas``. Other installations may have an unverified cached
version or runner and should inspect ``errors`` and ``tool`` instead of
assuming validity.

Versioned Scan Manifests
------------------------

Import ``ScanManifest`` from its data module:

.. code-block:: python

   from mcstasscript.data.scan_manifest import ScanManifest

Create a manifest only from a valid preflight report. It copies the plan,
instrument snapshot, effective settings, tool information, output root,
errors, and warnings. No point list is expanded:

.. code-block:: python

   report.raise_for_errors()
   manifest = ScanManifest.from_preflight(report)

The manifest is schema ``mcstasscript.scan_manifest`` version ``1``. It keeps
point records sparse, so an unobserved point is represented by the plan and
its aggregate count, not by a pending record for every point. Record an
observation by its canonical integer plan ID:

.. code-block:: python

   from pathlib import Path
   from mcstasscript.data.scan import ScanSource

   manifest.record_point(
       0,
       source=ScanSource(Path(report.output_path) / "0"),
       status="succeeded",
       data_status="available",
       seed={"intended": 42, "actual": 42},
       diagnostics=[],
   )

``record_point`` validates the point ID, source format, status, data status,
seed values, and JSON-safe diagnostics. A point can be recorded only once.
Source locations are passive locators; recording one does not inspect or load
the path. Process evidence is separate from point records:

.. code-block:: python

   manifest.record_execution(
       status="succeeded",
       outcome="completed",
       returncode=0,
       diagnostics=[],
   )

This updates process metadata only. It does not launch a process or change
point statuses. ``manifest.execution`` reports the process evidence, total
plan count, number of recorded points, and counts for the point statuses.

The metadata properties are detached from the manifest's internal state.
``fixed_parameters``, ``parameter_snapshot``, ``effective_settings``, and
``tool`` return fresh mappings. ``errors`` and ``warnings`` return fresh lists.
``points`` is the sparse-record view and is equivalent to ``records``;
``records`` has read-only outer and per-point mappings. ``execution`` returns a
fresh aggregate whose ``process`` mapping is also detached. Mutating any
returned metadata or process/point view therefore cannot rewrite the
manifest.

JSON And Portable Sources
~~~~~~~~~~~~~~~~~~~~~~~~~

``to_dict`` returns a complete JSON-safe document. ``from_dict`` validates the
schema, version, plan, sparse point records, source locators, and execution
aggregate before reconstructing a manifest:

.. code-block:: python

   import json

   encoded = json.dumps(manifest.to_dict(), allow_nan=False)
   restored = ScanManifest.from_dict(json.loads(encoded))

When a source is below ``output_root``, its serialized path is relative and
safe from traversal. An external source remains an explicit absolute path.
Relocate relative output by supplying a new root to ``resolve_source``:

.. code-block:: python

   relocated = restored.resolve_source(0, root="/moved/results")
   print(relocated.path)

Without ``root``, relative sources resolve against the manifest's stored
``output_root``. ``resolve_source`` returns a ``ScanSource`` locator and does
not require the target file to exist. A point without a source resolves to
``None``; an unrecorded point raises ``KeyError``.

The manifest API serializes metadata only. It does not automatically launch a
simulation or load detector data. ``instrument.scan`` is the separate execution
owner that writes a manifest and native output; its result-loading behavior is
described in :doc:`scan_execution` and :doc:`scan_results`.
