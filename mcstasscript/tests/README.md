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
