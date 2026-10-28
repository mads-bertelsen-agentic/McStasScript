"""List-compatible monitor results from a single simulation."""


class McStasDataCollection(list):
    """Monitor datasets from one simulation, with lookup and plotting helpers.

    Construction, indexing, mutation, and other list operations retain their
    usual behavior. Slicing, ``copy()``, concatenation, and repetition return
    ordinary lists; :meth:`find_all` returns a new collection. Datasets are not
    copied when constructing or searching a collection.

    Parameters
    ----------
    iterable : iterable of McStasData, optional
        Monitor datasets, in their original order.
    """

    def find(self, name=None, *, component=None, filename=None):
        """Return exactly one matching monitor dataset.

        Parameters
        ----------
        name : str, optional
            Match the dataset name first, falling back to its filename only
            when no dataset name matches.
        component : str, optional
            Match the dataset name exactly, without filename fallback.
        filename : str, optional
            Match the metadata filename exactly. May be combined with
            ``component`` to select an output from a multi-output monitor.

        Returns
        -------
        McStasData
            The matching dataset, not a copy.

        Raises
        ------
        TypeError
            No selector, a non-string selector, or both ``name`` and explicit
            selector keywords were supplied.
        NameError
            No dataset matches the selector.
        ValueError
            More than one dataset matches. Use an explicit filename or
            :meth:`find_all` instead.
        """
        matches = self.find_all(name, component=component, filename=filename)
        selector = (repr(name) if name is not None
                    else "component={!r}, filename={!r}".format(component, filename))
        if not matches:
            raise NameError("No dataset matching {} found.".format(selector))
        if len(matches) > 1:
            raise ValueError(
                "Multiple datasets match {}; use an explicit filename or "
                "find_all().".format(selector))
        return matches[0]

    def find_all(self, name=None, *, component=None, filename=None):
        """Return a collection of matching datasets, possibly empty.

        Selectors follow :meth:`find`: ``name`` uses dataset-name precedence
        and filename fallback, whereas ``component`` and ``filename`` are exact
        filters. Explicit filters may be combined with each other, but not with
        ``name``. Matches retain their original order and identity.
        """
        if name is None and component is None and filename is None:
            raise TypeError("Supply a name, component, or filename selector.")
        if name is not None and (component is not None or filename is not None):
            raise TypeError("Use either name or explicit component/filename selectors.")
        for label, value in (("name", name), ("component", component),
                             ("filename", filename)):
            if value is not None and not isinstance(value, str):
                raise TypeError("{} must be a string.".format(label))

        if name is not None:
            matches = McStasDataCollection(data for data in self if data.name == name)
            if matches:
                return matches
            return McStasDataCollection(
                data for data in self if data.metadata.filename == name)

        return McStasDataCollection(
            data for data in self
            if (component is None or data.name == component)
            and (filename is None or data.metadata.filename == filename))

    def plot(self, **kwargs):
        """Plot monitors in subplots using ``make_sub_plot(self, **kwargs)``.

        Options and return values are those of
        :func:`mcstasscript.make_sub_plot`, including ``return_fig_ax=True``.
        """
        from mcstasscript.interface.plotter import make_sub_plot

        return make_sub_plot(self, **kwargs)

    def plot_individual(self, **kwargs):
        """Plot monitors separately using ``make_plot(self, **kwargs)``.

        Options and return values are those of :func:`mcstasscript.make_plot`.
        A figure/axes pair is returned with ``return_fig_ax=True`` only when
        there is one plottable dataset, matching the existing plotter.
        """
        from mcstasscript.interface.plotter import make_plot

        return make_plot(self, **kwargs)
