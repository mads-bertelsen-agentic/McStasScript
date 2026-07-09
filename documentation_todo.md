# Documentation Build Workflow — Todo

## Completed

- [x] Create `improve-docs-workflow` branch from `master`
- [x] Fix `docs/requirements.txt` — add sphinx and myst-nb, remove unused nbsphinx
- [x] Rewrite `docs/source/conf.py` — remove hardcoded path, add autodoc_exclude_patterns, set `nb_execution_mode = 'off'`
- [x] Update `docs/Makefile` — default SPHINXBUILD to `python3 -m sphinx`
- [x] Update `docs/source/index.rst` — explicit public module list, libpyvinyl optional note
- [x] Create `docs/build_docs.sh` — build script with `--execute` flag (300s timeout)
- [x] Create `docs/source/_static/` directory
- [x] Clean stale `_autosummary/` stubs — removed 200+ old .rst files, added to `.gitignore`
- [x] Create micromamba environment (`docs/environment.yml`, env: `mcstasscript-docs`)
- [x] Install McStas from conda-forge in doc env (auto-detected, no manual config needed)
- [x] Remove hardcoded McStas paths from tutorial 1 notebook (cell removed)
- [x] Update installation notebook — note conda auto-configuration
- [x] Update `.gitignore` — doc build artifacts, simulation output patterns
- [x] Verify build succeeds — 33 pages, ~10s without `--execute`

## Remaining

### High priority

- [ ] Add `.github/workflows/build_docs.yml` CI action for automated doc builds

### Medium priority

- [ ] Fix 10 notebooks that fail with `--execute` (McStas 3.7 API changes):
  - `tutorial_3_EXTEND_and_WHEN` — `maxdiv_h`, `nh` removed from `DivPos_monitor`
  - `component_object` — duplicate component name `"code_arm"`
  - `instrument_object` — simulation exceeds 300s timeout
  - All Union tutorials — McStas 3.7 component parameter changes
  - `tutorial_5_MCPL_bridges` — needs investigation
- [ ] Address 79 cross-reference warnings (stale `_autosummary/` paths in notebooks)
- [ ] Add `make clean` target or document autosummary cleanup step

### Nice to have

- [ ] Pin sphinx/myst-nb versions in requirements.txt for reproducible builds
- [ ] Add RTD or GitHub Pages deployment for published docs
- [ ] Add linkcheck build (`sphinx-build -b linkcheck`)

## Notes

- **Without `--execute`**: build is fast (~10s), notebooks rendered from stored outputs
- **With `--execute`**: notebooks run live, 13/23 pass, 10 fail due to McStas 3.7 API changes
- Tutorial 1 now passes with conda McStas (was failing on hardcoded `/Applications/McStas-2.7.1.app/` path)
