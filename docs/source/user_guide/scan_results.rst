Scan Result Model
=================

This is the execution-free foundation for scan loading. The current milestone
provides result records and native ASCII/NeXus fixtures, not ``load_scan()`` or
``instrument.scan()``. Readers and execution will be added in that order.
No new dependencies are required.

Points, Identities, And Sources
-------------------------------

:class:`~mcstasscript.ScanResult` holds an ordered tuple of
:class:`~mcstasscript.ScanPoint` objects. A point has a stable integer or string
``point_id``, parameter values, optional source location, summary observations,
and independent execution and data statuses. Points with identical parameter
values remain separate measurements. Source order is preserved; a grid is never
inferred from the number of parameters or from Cartesian-looking observations.
If the original requested plan is unknown, ``scan.plan`` remains ``None``.

Low-level reader support records live in ``mcstasscript.data.scan``:

.. code-block:: python

   from mcstasscript import ScanPoint, ScanResult
   from mcstasscript.data.scan import (
       ScanMonitor, ScanMonitorSummary, ScanSource,
   )

   monitor = ScanMonitor("PSD", "PSD.dat")
   point = ScanPoint(
       point_id=7,
       parameters={"theta": 2.0, "sample": "vanadium"},
       source=ScanSource("existing_scan/7"),
       summaries=[ScanMonitorSummary(
           monitor, total_I=120.0, total_E=3.0,
           availability={"total_N": False},
           provenance={"total_I": "native summary", "total_E": "native summary"},
       )],
   )
   scan = ScanResult([point], parameter_names=["theta", "sample"])
   psd = scan.dataset("PSD")
   print(psd.total_I, psd.total_E, psd.coordinates)

``ScanSource`` is a passive locator, not a file handle or a loader. ASCII sources
identify point directories. NeXus sources identify a file and an explicit entry:
``ScanSource("scan/mccode.h5", format="nexus", entry="entry3")``. Relative file
paths are anchored at construction so later working-directory changes cannot
redirect loading. Entry numbers identify storage, not necessarily original point
IDs. Readers must verify their association with measurements.

``ScanMonitor`` captures a logical component/filename identity independently of
mutable display names. An optional discriminator distinguishes otherwise
identical outputs. ``scan.dataset()`` follows collection-style name precedence
and explicit ``component=`` / ``filename=`` selection, but resolves an identity
once across the whole scan. Missing instances do not fall back to another output
at individual points. Supply an exact ``ScanMonitor`` to select a discriminator.
An unknown filename is represented as ``None``, not guessed from another point.

Summary Availability
--------------------

A monitor view returns one value per point, retaining missing observations as
``NaN``. ``point_ids`` and ``coordinates`` retain canonical point order.
``valid`` is the finite-intensity mask, and ``validity("total_E")`` or
``validity("total_N")`` selects another quantity's mask.

``ScanMonitorSummary.availability`` distinguishes an observed value (``True``),
an inspected but absent quantity (``False``), and unknown availability (``None``).
An absent summary record is unknown, not evidence that a monitor was absent;
readers record inspected absence explicitly. The validated summary availability
mapping is read-only. An explicit zero is a valid observation, not missing data.
Availability is
independent of provenance, which records whether a total came from a native
summary, monitor metadata, or an explicitly documented derivation. Summary inputs
are detached from mutable metadata attached to full monitor datasets.

Full-array availability is separate. A reported total error does not prove that
an error array exists. In particular, compatibility zero-fills in ordinary
single-run readers must not be presented as known zero uncertainties by scans.
Readers populate the ``ScanPointData.availability`` sidecar before any such
fallback, without changing the existing single-run numeric results.

Lazy Data And Diagnostics
-------------------------

At this milestone, a reader supplies a ``loader(source)`` callback returning
``ScanPointData(data, diagnostics=..., availability=...)``. The model itself
performs no file discovery or parsing. ``data`` is a ``McStasDataCollection`` of
readable monitors; diagnostic and array-availability mappings use stable
``ScanMonitor`` keys. A point can alternatively be constructed with already loaded
``data=``. Summary access, monitor selection, and point inspection never trigger
the loader.

Explicit ``point.data`` access calls the loader once and returns the same cached
collection thereafter, including when it is empty. Partial results preserve
readable monitors, retain per-monitor diagnostics, and warn once. Entirely
unreadable results raise ``ScanDataError`` with the point ID and source. Expected
failures are cached rather than silently retried. ``point.clear_cache()`` drops
cached arrays and errors and explicitly permits a new load; it does not discard
summaries or change execution status. Cached arrays are shared mutable objects.

Callbacks must close their files and return detached data. They translate expected
format failures into ``ScanDataError``; ordinary I/O errors are also handled.
Unexpected programming errors propagate without being classified as failed
measurements. Automatic cache eviction, concurrent loading, and callback
serialization are not part of this milestone.

Execution Evidence
------------------

``point.status`` records execution evidence and normally stays ``"unknown"`` for
native files without an independent status record. ``point.data_status`` records
``"unknown"``, ``"available"``, ``"partial"``, ``"missing"``, or
``"unreadable"``. Successfully loading data does not establish execution success;
failed execution may also leave readable data. ``scan.execution`` stores aggregate
process evidence separately and never overwrites point status.

The native fixture provenance and capability matrix are documented in
``mcstasscript/tests/scan_fixtures/README.md``. They describe observed McStas
3.7.14 layouts and limitations, not a blanket guarantee for other releases.
