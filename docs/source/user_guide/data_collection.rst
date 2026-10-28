Single-Run Data Collections
===========================

``instrument.backengine()`` and ``mcstasscript.load_data(folder)`` return a
:class:`~mcstasscript.McStasDataCollection`: a list of the monitor datasets from
one simulation, with convenient lookup and plotting methods. ASCII and NeXus
results use the same container. Individual monitor datasets are unchanged.

Working With Existing Code
--------------------------

The collection inherits directly from ``list``. Existing indexing, iteration,
unpacking, equality, and list mutation continue to work:

.. code-block:: python

   import mcstasscript as ms

   data = ms.load_data("simulation_output")
   assert isinstance(data, list)
   first_monitor = data[0]
   for monitor in data:
       print(monitor.name)

Slicing, ``data.copy()``, concatenation, and repetition return ordinary lists,
just as they do for a standard list subclass. Use ``ms.McStasDataCollection(items)``
to wrap a list when lookup or plotting methods are needed. Construction does not
copy the contained monitor objects. ``copy.copy()``, ``copy.deepcopy()``, and
pickle preserve the collection class, following their normal copy semantics.

The concrete type has changed, so replace checks such as ``type(data) is list``
with ``isinstance(data, list)``. The collection does not replace libpyvinyl's
internal, key-based ``DataCollection``.

Looking Up Monitors
-------------------

.. code-block:: python

   psd = data.find("PSD")
   psd = data.find(filename="PSD.dat")
   psd = data.find(component="PSD", filename="PSD.dat")
   outputs = data.find_all(component="PSD")

The shorthand ``find("PSD")`` matches the current dataset name (``monitor.name``)
first. Only if no name matches does it try the metadata filename. Explicit
``component=`` and ``filename=`` filters perform exact matches without fallback;
they can be combined, but cannot be combined with the shorthand ``name`` argument.
Filenames must match the metadata filename, not an arbitrary absolute file path.

``find()`` always returns one dataset. It raises ``NameError`` when nothing
matches and ``ValueError`` when multiple datasets match. Select a filename to
disambiguate components that produce several outputs. ``find_all()`` always
returns a new collection, possibly empty, retaining the original dataset objects
and their order. Both methods require at least one string selector and reject
invalid selector combinations with ``TypeError``.

The legacy ``ms.name_search(name, data)`` function continues to accept both plain
lists and collections. Its return behavior is unchanged: a single dataset for
one match, a plain list for multiple matches, and an error for missing matches.
``ms.name_plot_options()`` continues to work with both containers as well.

Plotting
--------

.. code-block:: python

   data.plot()                    # One figure containing subplots
   data.plot_individual()         # Separate monitor figures
   figure, axes = data.plot(return_fig_ax=True)

``plot()`` delegates to :func:`~mcstasscript.make_sub_plot` and
``plot_individual()`` delegates to :func:`~mcstasscript.make_plot`. They retain
the existing plotters' keyword options, event-data filtering, plot-option mutation,
and return behavior. In particular, ``plot_individual(return_fig_ax=True)`` returns
a figure/axes pair only when there is one plottable monitor. Existing calls such
as ``ms.make_sub_plot(data)`` need no changes.

API Reference
-------------

.. autoclass:: mcstasscript.McStasDataCollection
   :members: find, find_all, plot, plot_individual
