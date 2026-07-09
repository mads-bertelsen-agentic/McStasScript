# Documentation Build Workflow — Todo

## Completed

- [x] Create `improve-docs-workflow` branch from `master`
- [x] Fix `docs/requirements.txt` — add sphinx and myst-nb, remove unused nbsphinx
- [x] Rewrite `docs/source/conf.py` — remove hardcoded path, add autodoc_exclude_patterns, set `nb_execution_mode = 'off'`
- [x] Update `docs/Makefile` — default SPHINXBUILD to `python3 -m sphinx`
- [x] Update `docs/source/index.rst` — explicit public module list, libpyvinyl optional note
- [x] Create `docs/build_docs.sh` — one-command build script with `--no-deps` flag
- [x] Create `docs/source/_static/` directory
- [x] Clean stale `_autosummary/` stubs — removed 200+ old .rst files
- [x] Verify build succeeds — 33 pages in ~1 min

## Remaining

### High priority

- [x] Create micromamba environment for local doc builds (`docs/environment.yml`, env: `mcstasscript-docs`)
- [ ] Add `docs/build/` and `docs/source/_autosummary/` to `.gitignore`
- [ ] Add `.github/workflows/build_docs.yml` CI action for automated doc builds

### Medium priority

- [ ] Address 79 cross-reference warnings in notebooks (stale `_autosummary/` paths)
- [ ] Add `make clean` target or document autosummary cleanup step

### Nice to have

- [ ] Pin sphinx/myst-nb versions in requirements.txt for reproducible builds
- [ ] Add RTD or GitHub Pages deployment for published docs
- [ ] Add linting check for broken links (`sphinx-build -b linkcheck`)
