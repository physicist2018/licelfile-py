# Changelog

## [1.5.0] — 2026-06-09

### Added
- `LicelPack.average()` — creates a new `LicelFile` with element-wise averaged
  profiles across all files in the pack
- `LicelPack.to_npz()` / `LicelPack.from_npz()` — save/load pack to/from
  compressed NumPy `.npz` archive (structured arrays + NaN-padded data matrix)
- `to_netcdf()` / `from_netcdf()` — optional NetCDF I/O in CF-1.8 conventions
  (requires `netCDF4`, installable via `pip install licelformat[netcdf]`)
- New module `licelformat/ionetcdf.py` with NetCDF save/load functions
- Optional dependency group `[netcdf]` in `pyproject.toml`
- 14 tests for `LicelFile.glue()` + `LicelPack.glue()`
  (creation, zone verification, overwrite, missing channels, error handling,
  pack-level in-place mutation, failed file removal)
- 5 tests for `LicelPack.average()`
  (two files, three files, empty pack, mismatched profiles, truncation)

### Changed
- **`LicelFile.glue()`** — if a glued profile (`DeviceID == 'BG'`) with the
  same `wavelength`/`polarization` already exists, it is **updated in-place**
  instead of duplicating. `NDatasets` is not incremented on overwrite.
- **`LicelPack.glue()`** — now **mutates** `self` in-place (removes failed
  files, recomputes time range) and returns `self` instead of creating a new
  `LicelPack`. Old code that expected a new pack object needs updating.

### Documentation
- `API.md` — full documentation for `to_npz()`, `from_npz()`, `to_netcdf()`,
  `from_netcdf()`, updated glue docs
- `README.md` — new features section, code examples for `.npz`, NetCDF,
  updated glue example
- `CHANGELOG.md` created

## [1.4.4] — 2026-06-09

### Added
- CI workflow via GitHub Actions (`python-package.yml`)

## [1.4.0] — 2026-06-09

### Added
- `LicelPack.average()` — channel-wise average across files
- Properties `isPhoton`, `isAnalog`, `isGlued` on `LicelProfile`
- `truncate(rmax)` method on all three classes
- `subtract_background()` on all three classes (mean, median, dark modes)
- `LicelPack.filter_files()` — filter by file predicate, returns new `LicelPack`

### Changed
- `LicelPack.filter()` signature changed: now returns `LicelProfilesList`
  (flat list of profiles), old behavior moved to `filter_files()`
- `LicelFile.glue()` — uses numpy arrays for safe numeric computation
- Version bump from 0.1.1 → 1.0.0 → 1.4.0

## [0.1.1] — 2026-06-09

### Added
- Initial release as Python port of Go `licelfile` package
- `LicelProfile` — header parsing, metadata round-trip, scaling
- `LicelFile` — load/save, profile selection, wavelength filtering
- `LicelPack` — multi-file loading via glob masks and ZIP archives
- `filter()` — profile filtering by predicate
- `select_certain_wavelength()` — profile selection by wavelength/type
- `save_to_zip()` — export pack to ZIP archive
- `to_dict()` — JSON serialization for all classes
