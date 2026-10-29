"""Execution-free scan records shared by future ASCII and NeXus readers.

This module does not discover files, infer grids, or launch simulations. Point
readers supply detached summaries and a lazy loader returning :class:`ScanPointData`.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import numbers
import os
from pathlib import Path
from types import MappingProxyType
from typing import Optional
import warnings

import numpy as np

from .collection import McStasDataCollection


_TOTALS = ("total_I", "total_E", "total_N")
_DATA_STATUSES = ("unknown", "available", "partial", "missing", "unreadable")


@dataclass(frozen=True)
class ScanSource:
    """Passive point location, with no open files or loading policy.

    ``format`` is ``"ascii"`` for a point directory or ``"nexus"`` for an HDF5
    file and an explicit ``entry``. Relative paths are anchored at construction,
    so changing the working directory cannot redirect subsequent lazy loading.
    An entry identifies storage, not the original scan point ID.
    """

    path: Path
    format: str = "ascii"
    entry: Optional[str] = None

    def __post_init__(self):
        path = os.fspath(self.path)
        if "\x00" in os.fsdecode(path):
            raise ValueError("Scan source paths cannot contain NUL characters.")
        object.__setattr__(self, "path", Path(os.path.abspath(path)))
        if self.format not in ("ascii", "nexus"):
            raise ValueError("Scan source format must be 'ascii' or 'nexus'.")
        if self.format == "ascii" and self.entry is not None:
            raise ValueError("An ASCII source is a directory, not a NeXus entry.")
        if self.format == "nexus":
            if not isinstance(self.entry, str) or not self.entry.strip("/"):
                raise ValueError("A NeXus source needs an explicit entry path.")
            if "\x00" in self.entry:
                raise ValueError("NeXus entry paths cannot contain NUL characters.")
            if any(part in ("", ".", "..") for part in self.entry.strip("/").split("/")):
                raise ValueError("Invalid NeXus entry path.")
            object.__setattr__(self, "entry", "/" + self.entry.strip("/"))

    def __str__(self):
        return str(self.path) + ("::" + self.entry if self.entry else "")


@dataclass(frozen=True)
class ScanMonitor:
    """Stable logical output identity, independent of mutable dataset names.

    ``filename=None`` permits a summary-only identity when the source does not
    establish a filename. Readers must not guess a match to full data in that
    case. ``discriminator`` distinguishes otherwise identical output identities;
    source-specific HDF5 group names belong in reader metadata, not this key.
    """

    component: str
    filename: Optional[str] = None
    discriminator: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.component, str) or not self.component:
            raise ValueError("A monitor needs a nonempty component name.")
        for value in (self.filename, self.discriminator):
            if value is not None and not isinstance(value, str):
                raise TypeError("Monitor filename and discriminator must be strings or None.")


@dataclass(frozen=True)
class ScanMonitorSummary:
    """Detached scalar observations for one monitor at one point.

    ``availability`` maps total names to True (observed/explicitly derived),
    False (inspected and absent), or None (unknown). A supplied numeric total
    implies True; an omitted total defaults to unknown, never a known zero.
    ``provenance`` records how each total was obtained, separately from the
    availability of full detector arrays in :class:`ScanPointData`.
    """

    monitor: ScanMonitor
    total_I: Optional[float] = None
    total_E: Optional[float] = None
    total_N: Optional[float] = None
    availability: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.monitor, ScanMonitor):
            raise TypeError("Summary monitor must be a ScanMonitor.")
        if set(self.availability) - set(_TOTALS):
            raise ValueError("Summary availability keys must be total_I, total_E, or total_N.")
        available = {}
        for quantity in _TOTALS:
            value = getattr(self, quantity)
            if value is not None and (isinstance(value, bool) or
                                      not isinstance(value, numbers.Real)):
                raise TypeError("{} must be a real number or None.".format(quantity))
            state = self.availability.get(quantity, True if value is not None else None)
            if state is not None and not isinstance(state, bool):
                raise TypeError("Availability must be True, False, or None.")
            if (value is not None and state is not True) or (value is None and state is True):
                raise ValueError("{} value conflicts with its availability.".format(quantity))
            available[quantity] = state
        object.__setattr__(self, "availability", MappingProxyType(available))
        object.__setattr__(self, "provenance", deepcopy(self.provenance))
        object.__setattr__(self, "metadata", deepcopy(self.metadata))

    def __reduce__(self):
        # Reconstruct the validated record, not its unpicklable mapping proxy.
        return (type(self), (self.monitor, self.total_I, self.total_E, self.total_N,
                             dict(self.availability), self.provenance, self.metadata))


@dataclass(frozen=True)
class ScanPointData:
    """Reader report for a lazy point load, including partial-monitor issues.

    ``data`` contains readable monitors in source order. ``diagnostics`` maps
    stable ScanMonitor identities to diagnostic details. ``availability`` is a
    scan-owned sidecar keyed by those same identities: its per-monitor mappings
    describe full-array quantities, e.g. ``{"Error": False}``. Readers capture
    this before legacy zero-filling; it is independent of summary availability.
    An empty collection with no diagnostics is a valid, cached empty result.
    """

    data: McStasDataCollection
    diagnostics: dict = field(default_factory=dict)
    availability: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.data, list):
            raise TypeError("Point data must be a list-compatible monitor collection.")
        if not isinstance(self.data, McStasDataCollection):
            object.__setattr__(self, "data", McStasDataCollection(self.data))
        for mapping in (self.diagnostics, self.availability):
            if any(not isinstance(key, ScanMonitor) for key in mapping):
                raise TypeError("Point diagnostics and availability need ScanMonitor keys.")
        for quantities in self.availability.values():
            if any(not isinstance(name, str) or (state is not None and not isinstance(state, bool))
                   for name, state in quantities.items()):
                raise TypeError("Array availability maps quantity names to True, False, or None.")
        object.__setattr__(self, "diagnostics", deepcopy(self.diagnostics))
        object.__setattr__(self, "availability", deepcopy(self.availability))


class ScanDataError(RuntimeError):
    """Unavailable output, optionally retaining per-monitor diagnostic sidecars."""

    def __init__(self, message, data_status="unreadable", *, diagnostics=None,
                 availability=None):
        if data_status not in ("unknown", "missing", "unreadable"):
            raise ValueError("Invalid data status for a point-loading error.")
        super().__init__(message)
        self.data_status = data_status
        self.diagnostics = deepcopy(diagnostics) if diagnostics is not None else None
        self.availability = deepcopy(availability) if availability is not None else None


class ScanPoint:
    """One measurement with stable identity, summaries, and optional full data.

    ``loader(source)`` must return a detached ScanPointData report and close any
    files before returning. It runs only on explicit ``data`` access. Successful,
    partial, empty, and expected failed loads are cached until ``clear_cache()``.
    Readers translate expected parse failures to ScanDataError. Programming
    errors in a loader are not translated into data failures.

    ``status`` describes execution evidence, not readable output. Native-file
    imports normally leave it unknown; ``data_status`` changes during loading
    without changing execution status. Parameter and metadata inputs are copied.
    """

    def __init__(self, point_id, parameters, source=None, *, summaries=(),
                 status="unknown", data_status="unknown", metadata=None,
                 loader=None, data=None):
        if isinstance(point_id, bool) or not isinstance(point_id, (numbers.Integral, str)):
            raise TypeError("Point ID must be an integer or string.")
        if point_id == "" or (isinstance(point_id, numbers.Integral) and point_id < 0):
            raise ValueError("Point ID must be nonnegative or a nonempty string.")
        if source is not None and not isinstance(source, ScanSource):
            raise TypeError("Point source must be a ScanSource or None.")
        if status not in ("unknown", "pending", "running", "succeeded", "failed",
                          "interrupted", "skipped"):
            raise ValueError("Invalid point execution status.")
        if data_status not in _DATA_STATUSES:
            raise ValueError("Invalid point data status.")
        if loader is not None and not callable(loader):
            raise TypeError("Point loader must be callable.")
        self._point_id = point_id
        self.parameters = deepcopy(dict(parameters))
        self.source = source
        self.status = status
        self.data_status = data_status
        self.metadata = deepcopy(metadata) if metadata is not None else {}
        self.summaries = {}
        for summary in summaries:
            if not isinstance(summary, ScanMonitorSummary):
                raise TypeError("Point summaries must be ScanMonitorSummary records.")
            if summary.monitor in self.summaries:
                raise ValueError("Duplicate monitor identity at point {!r}.".format(point_id))
            self.summaries[summary.monitor] = summary
        self._loader = loader
        self._loaded = None
        self._load_error = None
        self.diagnostics = {}
        self.array_availability = {}
        if data is not None:
            try:
                self._cache_data(data if isinstance(data, ScanPointData) else ScanPointData(data))
            except ScanDataError as error:
                self._cache_error(error)

    @property
    def point_id(self):
        """Stable measurement ID; unrelated to coordinates or list position."""
        return self._point_id

    @property
    def is_loaded(self):
        """Whether full point data has been successfully cached, including empty data."""
        return self._loaded is not None

    def _cache_data(self, report):
        if not isinstance(report, ScanPointData):
            raise TypeError("Point loader must return ScanPointData.")
        self.diagnostics = deepcopy(report.diagnostics)
        self.array_availability = deepcopy(report.availability)
        if report.diagnostics and not report.data:
            raise ScanDataError("No readable monitors: {}".format(report.diagnostics),
                                diagnostics=report.diagnostics,
                                availability=report.availability)
        self._loaded = report.data
        self.data_status = "partial" if report.diagnostics else "available"
        if report.diagnostics:
            warnings.warn("Partial data for scan point {!r} from {}: {}".format(
                self.point_id, self.source, report.diagnostics), UserWarning, stacklevel=3)

    def _cache_error(self, error):
        if isinstance(error, ScanDataError):
            if error.diagnostics is not None:
                self.diagnostics = deepcopy(error.diagnostics)
            if error.availability is not None:
                self.array_availability = deepcopy(error.availability)
        state = (error.data_status if isinstance(error, ScanDataError) else
                 "missing" if isinstance(error, FileNotFoundError) else "unreadable")
        message = "Cannot load scan point {!r} from {}: {}".format(
            self.point_id, self.source, error)
        self.data_status = state
        self._load_error = (message, state)
        return ScanDataError(message, state, diagnostics=self.diagnostics,
                             availability=self.array_availability)

    @property
    def data(self):
        """Shared cached monitor collection; no implicit retries or file handles."""
        if self._loaded is not None:
            return self._loaded
        if self._load_error is not None:
            raise ScanDataError(*self._load_error, diagnostics=self.diagnostics,
                                availability=self.array_availability)
        try:
            if self._loader is None:
                state = "missing" if self.source is None else self.data_status
                if state not in ("missing", "unreadable"):
                    state = "unknown"
                raise ScanDataError("No full-data loader is available.", state)
            self._cache_data(self._loader(self.source))
        except (ScanDataError, OSError) as error:
            raise self._cache_error(error) from error
        return self._loaded

    def clear_cache(self):
        """Explicitly permit a reload without changing summaries or execution status."""
        self._loaded = None
        self._load_error = None
        self.diagnostics = {}
        self.array_availability = {}
        self.data_status = "unknown"


class ScanResult:
    """Ordered measurement records, with no inferred grid or execution behavior.

    Readers supply points in verified source order. An absent ``plan`` stays
    absent: observed Cartesian-looking coordinates do not establish scan intent.
    ``execution`` holds aggregate process evidence separately from point status.
    """

    def __init__(self, points, *, parameter_names=None, plan=None, metadata=None,
                 execution=None):
        self.points = tuple(points)
        if any(not isinstance(point, ScanPoint) for point in self.points):
            raise TypeError("Scan results contain ScanPoint records.")
        ids = [point.point_id for point in self.points]
        if len(set(ids)) != len(ids):
            raise ValueError("Scan point IDs must be unique, even at repeated coordinates.")
        if parameter_names is None:
            parameter_names = dict.fromkeys(name for point in self.points
                                           for name in point.parameters)
        if isinstance(parameter_names, str):
            raise TypeError("Parameter names must be a sequence, not a string.")
        self.parameter_names = tuple(parameter_names)
        if any(not isinstance(name, str) for name in self.parameter_names):
            raise TypeError("Parameter names must be strings.")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("Scan parameter names must be unique.")
        self.plan = plan
        self.metadata = deepcopy(metadata) if metadata is not None else {}
        self.execution = deepcopy(execution) if execution is not None else {}

    @property
    def monitors(self):
        """Known logical monitor identities in first-observed order; no I/O."""
        return tuple(dict.fromkeys(key for point in self.points for key in point.summaries))

    @property
    def coordinates(self):
        """Point-ordered object arrays preserving strings, numbers, and missing values."""
        coordinates = {}
        for name in self.parameter_names:
            values = np.empty(len(self.points), dtype=object)
            for index, point in enumerate(self.points):
                values[index] = point.parameters.get(name)
            coordinates[name] = values
        return coordinates

    def dataset(self, name=None, *, component=None, filename=None):
        """Resolve one monitor identity scan-wide, then return a summary-only view.

        Uses the collection selector rules, but resolves once across the scan.
        A missing instance at a point never falls back to a different monitor.
        An exact ScanMonitor may be supplied instead of a string, including its
        discriminator. Ambiguous string selectors require explicit filters.
        """
        if name is None and component is None and filename is None:
            raise TypeError("Supply a name, component, or filename selector.")
        if name is not None and (component is not None or filename is not None):
            raise TypeError("Use either name or explicit component/filename selectors.")
        if isinstance(name, ScanMonitor):
            if name not in self.monitors:
                raise NameError("No scan monitor matches the identity.")
            return ScanMonitorView(self, name)
        if any(value is not None and not isinstance(value, str)
               for value in (name, component, filename)):
            raise TypeError("Monitor selectors must be strings.")
        if name is not None:
            matches = [key for key in self.monitors if key.component == name]
            if not matches:
                matches = [key for key in self.monitors if key.filename == name]
        else:
            matches = [key for key in self.monitors
                       if (component is None or key.component == component)
                       and (filename is None or key.filename == filename)]
        if not matches:
            raise NameError("No scan monitor matches the selector.")
        if len(matches) != 1:
            raise ValueError("Ambiguous scan monitor; use explicit component/filename filters.")
        return ScanMonitorView(self, matches[0])


class ScanMonitorView:
    """Point-aligned scalar monitor observations, without loading detector arrays."""

    def __init__(self, scan, monitor):
        self.scan = scan
        self.monitor = monitor

    @property
    def point_ids(self):
        return np.asarray([point.point_id for point in self.scan.points], dtype=object)

    @property
    def coordinates(self):
        return self.scan.coordinates

    def _values(self, quantity):
        return np.asarray([getattr(point.summaries[self.monitor], quantity)
                           if self.monitor in point.summaries and
                           getattr(point.summaries[self.monitor], quantity) is not None
                           else np.nan for point in self.scan.points], dtype=float)

    @property
    def total_I(self):
        return self._values("total_I")

    @property
    def total_E(self):
        return self._values("total_E")

    @property
    def total_N(self):
        return self._values("total_N")

    @property
    def availability(self):
        """Per-total True/False/None arrays, distinct from finite-value masks."""
        return {quantity: np.asarray([
            point.summaries[self.monitor].availability[quantity]
            if self.monitor in point.summaries else None
            for point in self.scan.points], dtype=object) for quantity in _TOTALS}

    def validity(self, quantity="total_I"):
        """Finite observations for a total, without treating missing values as zero."""
        if quantity not in _TOTALS:
            raise ValueError("Quantity must be total_I, total_E, or total_N.")
        observed = np.asarray([state is True for state in self.availability[quantity]])
        return np.isfinite(self._values(quantity)) & observed

    @property
    def valid(self):
        return self.validity("total_I")
