# Unit Tests

These are the base unit tests for McStasScript. They use the regular package
dependencies, including `ipympl`, and must not require the optional
`pythreejs` geometry-viewer dependency.

Run them with:

```bash
python -m pytest mcstasscript/tests/
```

The scan-result tests use callbacks and small native captures in `scan_fixtures/`.
Fixture verification requires only the regular dependencies, not McStas. See
`scan_fixtures/README.md` for native format capabilities and regeneration commands;
regenerating captures is separate from running the unit tests.

The native ASCII and NeXus reader tests include McStas 3.7.14 and 3.10.0
outputs, summary-only input, malformed/partial monitors, mixed-format
diagnostics, lazy HDF5 file lifetime, and declared parameter types. The
`scan_fixtures/mcstas_3_10/` verifier runs as part of this base suite without
invoking McStas. Live scan/reload checks belong to `integration_tests/`.
