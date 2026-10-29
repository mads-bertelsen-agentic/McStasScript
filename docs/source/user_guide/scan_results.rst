Scan Result Model
=================

This page describes the execution-free scan-result model. Native ASCII and
NeXus output can be loaded without starting McStas, running a subprocess, or
requiring an instrument object. Loading observes the files that exist; it does
not reconstruct an execution plan.
No new dependencies are required. Scan execution and overview plotting are
separate, later features.

For runnable examples that write an instrument, run native ``!mcrun`` scans,
and read their output, see :doc:`../tutorial/McStasScript_tutorial_7_native_scans`.

Loading Native Scans
--------------------

Use the public ``ms.load_scan`` entry point for a scan directory, an ASCII
summary, or a NeXus file:

.. code-block:: python

   import mcstasscript as ms

   ascii_scan = ms.load_scan("ascii_scan")
   ascii_summary = ms.load_scan("ascii_scan/mccode.dat")
   custom_summary = ms.load_scan("ascii_scan/summary.results")
   combined_file = ms.load_scan("nexus_scan/mccode.h5")
   combined_directory = ms.load_scan("nexus_scan")
   old_per_point = ms.load_scan("old_nexus_scan")  # 0/mccode.h5, 1/mccode.h5, ...

Any non-HDF5 file supplied as a path is read as an ASCII summary, including a
custom extension. A directory containing one combined ``mccode.h5`` is loaded
as combined NeXus; a directory containing numbered per-point HDF5 files uses
those point directories. Summary-only points provide scalar monitor results
but have no full-data source.

Points, Identities, And Parameters
----------------------------------

``ScanResult.points`` is the ordered collection of observed
``ScanPoint`` records. Numbered ASCII directories, and legacy per-point NeXus
directories such as ``0/mccode.h5``, have known numeric IDs from their
directory names; their ``scan.metadata['point_id_basis']`` is
``'numbered directories'``. Summary-only ASCII input uses an encoded row ID or
summary row order when no directory ID exists.

Combined NeXus assigns zero-based integer IDs in storage order. These IDs are
synthetic collection IDs, not the one-based ``entry1`` names and not original
plan IDs. For this layout
``scan.metadata['point_id_basis'] == 'entry order'``. If a failed point has no
entry, later entries are compacted in storage order; the loader cannot map them
back to the requested plan. ``scan.plan`` remains ``None``. Never infer a
requested grid or point count from coordinates, entry names, headers, or
``Numpoints``.

Native parameter declarations are used when present. Declared integer, floating
point, and string parameters retain their numeric or string types; values such
as a declared string ``001`` are not silently converted to an integer. Point
metadata is authoritative when a native summary header orders coordinates
differently. Repeated parameter values remain separate points.

Summary Association
-------------------

Use a summary view without loading detector arrays:

.. code-block:: python

   spectrum = ascii_scan.dataset("spectrum")
   print(spectrum.total_I, spectrum.coordinates)
   point_data = ascii_scan.points[0].data  # first full-data access; then cached

Summary rows attach only when their point identity or parameter coordinates give
an unambiguous match. Ambiguous, conflicting, or duplicate-coordinate evidence
is not used to guess an association. Rows that cannot be safely attached are
either retained as summary-only records or remain, with their raw contents, in
``scan.metadata['unassociated_summary_rows']``; the reason is retained in
``scan.metadata['diagnostics']``. Summary-only monitor metadata includes the
native raw row and line number.

Categorical summaries can contain numeric category codes instead of labels.
For a summary-only point, those codes remain in
``point.metadata['encoded_parameters']`` and unavailable names are listed in
``point.metadata['unavailable_parameters']``. The loader never invents labels
or reconstructs missing coordinates from a code.
Legacy whole-row indices are retained as ``point.metadata['encoded_point_id']``.

Native headers and declared counts are advisory. ASCII variable order can be
hash-dependent or differ from row order, and native ``Numpoints`` can be a
per-axis count or describe a complete plan when rows are missing. Point
metadata and observed rows are retained instead of being corrected to fit a
header.

Mixed ASCII And NeXus Output
----------------------------

A root ASCII summary alongside a combined ``mccode.h5`` is supported: scalar
summary values and NeXus arrays are merged only after point identity and total
values are validated. If numbered directories contain full ASCII detector files
as well as NeXus output, loading the directory rejects the competing full
sources. Select the NeXus file explicitly, for example
``ms.load_scan("mixed/mccode.h5")``.

Numbered ASCII directories that contain metadata-only ``mccode.sim`` files can
be merged with NeXus arrays. The association must be unique; repeated
coordinates need a distinguishing seed, and conflicts or ambiguity are
rejected rather than silently combined.

Execution, Summaries, And Data
------------------------------

Native files do not provide an independent execution-status record. Therefore
native loads have ``scan.execution == {}`` and leave ``point.status`` as
``"unknown"``. ``point.data_status`` separately reports full-data readability
(``"unknown"``, ``"available"``, ``"partial"``, ``"missing"``, or
``"unreadable"``); readable data does not prove execution success.

Summary totals and full detector-array availability are separate. A monitor
view represents missing totals as ``NaN`` and exposes a True/False/None
``availability`` state. Full-data readers populate
``point.array_availability`` and per-monitor diagnostics before legacy loaders
can zero-fill missing ``Error`` or ``Ncount`` arrays; a compatibility zero is
not evidence of an observed zero uncertainty or count.

Point data is lazy. Accessing ``point.data`` invokes its detached loader once
and caches successful, partial, empty, and expected-error results. Partial
loads retain readable monitors and per-monitor diagnostics. An entirely
unreadable point raises ``ScanDataError`` with the point and source context;
expected errors are not silently retried. ``point.clear_cache()`` explicitly
permits another load without discarding summaries or changing execution status.
Summary access and monitor selection do not trigger array loading.

Fixture-Version Scope
---------------------

The retained native fixtures show version-specific behavior rather than a
portable scan-format contract:

* **Seeds:** McStas 3.7.14 reuses base seed ``314159`` for sequential points,
  while parallel points use ``314159 + 1024 * point_index``. McStas 3.10.0
  applies the point-index seed offset to both sequential and parallel points.
* **Categorical summaries:** In 3.7.14, a nonnumeric parameter can replace the
  whole summary coordinate prefix with a zero-based row index; point metadata
  remains authoritative. In 3.10.0, summaries contain per-list numeric category
  indices while labels remain in point metadata, and repeated coordinates stay
  distinct.
* **Partial runs:** The 3.7.14 ASCII fixture retains failed-point metadata but
  has no later point outputs; its NeXus fixture retains failed-point metadata
  without full data. In 3.10.0, ASCII continues after the failed point, while
  combined NeXus contains entries for successful points only, so omitted plan
  IDs remain unknown.
* **Parallel output:** 3.7.14 NeXus split output uses separate
  ``0/mccode.h5`` and ``1/mccode.h5`` files. 3.10.0 merges split output into a
  root ``mccode.h5``; it also supports unequal comma-separated ``-N`` counts
  and list products through ``-L -M``, and fixes the ``--scan_split=2 -L``
  summary failure. The single ``-M -N N`` form still declares a per-axis
  count.

See ``mcstasscript/tests/scan_fixtures/README.md`` for fixture provenance and
the capability matrix, and ``mcstasscript/tests/scan_fixtures/mcstas_3_10/README.md``
for the McStas 3.10.0 compatibility probe. These are
observations of the captured McStas versions, not universal guarantees for
other McStas releases or McXtrace.
