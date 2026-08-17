# Add Test Plan

## Findings

- Instrument generation is in `mcstasscript/interface/instr.py`.
- `write_full_instrument()` writes `* %Parameters` at line 2108.
- Simulation results identify monitors through `data.name` and `data.metadata.total_I`.
- `mctest` parses lines matching `%Example: <parameters> Detector: <monitor>_I=<value>`.
- Unit tests belong in `mcstasscript/tests/test_Instr.py`.
- The instrument documentation is `docs/source/user_guide/instrument_object.ipynb`.

## Implementation

1. Add `_test_list = []` to the guarded initialization block in `McCode_instr.__init__`.
2. Implement `add_test(Name_of_monitor, intensity=None, included_pars=None)`:
   - Validate the monitor component exists.
   - Snapshot selected parameter names and current values.
   - Require selected parameters to have values.
   - If `intensity` is omitted, run `backengine()` and extract `metadata.total_I` for the named monitor.
   - Store plain data only, avoiding mutable parameter references.
3. Implement `show_tests()`:
   - Print each generated `%Example` line with a zero-based index.
   - Print `No instrument tests available` when empty.
4. Implement `remove_test(index)` using the displayed index.
5. Insert each stored line immediately before `* %Parameters` in `write_full_instrument()`.
6. Add docstrings and method-list entries describing the McStas `%Example` structure.
7. Add unit tests covering explicit intensity, parameter filtering, automatic intensity extraction, display, file placement, and indexed removal.
8. Add an integration test that builds a minimal `Source_simple` plus `PSD_monitor` instrument, adds an explicit expected test intensity, writes the instrument, and runs `mctest --local ...` instead of `backengine`.
9. Document the feature in the instrument-object notebook, including an emitted `%Example` line and `add_test`, `show_tests`, and `remove_test` usage.

## Assumptions

- Test indices are zero-based.
- `included_pars` only controls the recorded example parameters.
- Automatic intensity uses the monitor's total intensity (`metadata.total_I`), matching `mctest`.
