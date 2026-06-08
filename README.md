# licelformat

[![PyPI version](https://img.shields.io/pypi/v/licelformat.svg)](https://pypi.org/project/licelformat/)
[![Python](https://img.shields.io/pypi/pyversions/licelformat.svg)](https://pypi.org/project/licelformat/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

Parser for **Licel** binary format files — a common data format in atmospheric
lidar remote sensing. Reads analog and photon‑counting measurement profiles,
extracts metadata, and provides scaling/unscaling consistent with the
reference Go implementation.

## Features

- **Parse** Licel `.dat`/`.licel` files with full header + binary profile extraction
- **Scale** raw ADC counts → millivolts and raw photon counts → MHz
- **Round‑trip**: save → reload preserves data losslessly (unscale on write)
- **Multi‑file** loading via glob masks and ZIP archives
- **Filter** profiles by any predicate (`LicelFile.filter`) and files by any criteria (`LicelPack.filter`)
- **Save to ZIP** with configurable compression method and level
- **Profile selection** by wavelength and channel type
- **NumPy** arrays for all profile data — ready for further analysis

## Installation

```bash
pip install licelformat
```

## Quick start

### Load a single Licel file

```python
from licelformat import LoadLicelFile

lf = LoadLicelFile("path/to/file.DAT")

print(lf.MeasurementSite)       # "Vladivos"
print(lf.MeasurementStartTime)  # datetime(2020, 2, 10, 19, 22, 35)
print(len(lf.Profiles))         # 12

# First profile
p = lf.Profiles[0]
print(p.Wavelength)   # 355.0
print(p.Photon)       # False  (analog channel)
print(p.NShots)       # 2001
print(p.Data[:5])     # numpy float64 array, units: mV or MHz
```

### Select profiles by wavelength

```python
# Pick the 355 nm photon‑counting channel
profile = lf.select_certain_wavelength(is_photon=True, wavelength=355.0)
```

### Batch processing with LicelPack

```python
from licelformat import NewLicelPack

pack = NewLicelPack("/data/2020/*.DAT")
profiles = pack.select_certain_wavelength(is_photon=True, wavelength=532.0)

for p in profiles:
    print(p.Data.mean())
```

### Filter profiles in a LicelFile

```python
# Only photon‑counting channels
photon = lf.filter(lambda p: p.Photon)

# Only 532 nm channels
at_532 = lf.filter(lambda p: p.Wavelength == 532.0)
```

### Filter files in a LicelPack

```python
# Keep only files from a specific site
site_pack = pack.filter(lambda lf: lf.MeasurementSite == "Vladivos")

# Keep only files with profiles at 355 nm
has_355 = pack.filter(lambda lf: lf.select_certain_wavelength(True, 355.0).Wavelength != 0)

print(site_pack.StartTime, site_pack.StopTime)  # recomputed from filtered set
```

### Load from a ZIP archive

```python
from licelformat import NewLicelPackFromZip

pack = NewLicelPackFromZip("archive.zip")
```

### Save to a ZIP archive

```python
from licelformat import NewLicelPack

pack = NewLicelPack("/data/2020/*.DAT")

# Save with default compression (DEFLATED, level 6)
pack.save_to_zip("output.zip")

# Store without compression
import zipfile
pack.save_to_zip("output.zip", compression=zipfile.ZIP_STORED)

# Maximum compression
pack.save_to_zip("output.zip", compresslevel=9)
```

## Data scaling

Raw `int32` values stored on disk are automatically scaled during loading:

| Channel type | Scale factor                          | Result unit |
|-------------|---------------------------------------|-------------|
| Analog      | `DiscrLevel × 1000 / (2^AdcBits × NShots)` | mV          |
| Photon      | `1 / (NShots × 0.05)`                 | MHz         |

When saving, the inverse scaling is applied (`round(scaled / scale)`), so
round‑trip is **lossless**.

## API reference

### `licelformat.LicelProfile`

| Field          | Type      | Description                    |
|---------------|-----------|--------------------------------|
| `Active`      | `bool`    | Channel active flag            |
| `Photon`      | `bool`    | `True` for photon counting     |
| `LaserType`   | `int`     | Laser index (1, 2, 3)          |
| `NDataPoints` | `int`     | Number of bins                 |
| `Reserved`    | `list`    | 3 reserved values              |
| `HighVoltage` | `int`     | PMT high voltage (V)          |
| `BinWidth`    | `float`   | Range bin width (m)            |
| `Wavelength`  | `float`   | Laser wavelength (nm)          |
| `Polarization`| `str`     | "o", "s", or "p"                |
| `BinShift`    | `int`     | Bin shift                      |
| `DecBinShift` | `int`     | Decimal bin shift              |
| `AdcBits`     | `int`     | ADC resolution (bits)          |
| `NShots`      | `int`     | Number of laser shots          |
| `DiscrLevel`  | `float`   | Discriminator level (mV)      |
| `DeviceID`    | `str`     | 2‑char device ID               |
| `NCrate`      | `int`     | Crate slot number              |
| `Data`        | `np.ndarray` | Scaled float64 profile data  |

| Property      | Type      | Description                    |
|---------------|-----------|--------------------------------|
| `isPhoton`    | `bool`    | `True` if `DeviceID == 'BC'`   |
| `isAnalog`    | `bool`    | `True` if `DeviceID == 'BT'`   |
| `isGlued`     | `bool`    | `True` if `DeviceID == 'BG'`   |

Methods: `metadata()`, `profile()`, `scale_factor()`, `to_dict()`

### `licelformat.LicelFile`

Fields: `MeasurementSite`, `MeasurementStartTime`, `MeasurementStopTime`,
`AltitudeAboveSeaLevel`, `Longitude`, `Latitude`, `Zenith`, `Laser1NShots`,
`Laser1Freq`, `Laser2NShots`, `Laser2Freq`, `NDatasets`, `Laser3NShots`,
`Laser3Freq`, `FileLoaded`, `Profiles`

Methods: `select_certain_wavelength()`, `filter()`, `save()`, `to_bytes()`, `to_dict()`

### `licelformat.LicelPack`

Fields: `StartTime`, `StopTime`, `Data` (dict of `LicelFile`)

Methods: `select_certain_wavelength()`, `filter()`, `save()`, `save_to_zip()`, `to_dict()`

### Module‑level functions

| Function                    | Description                              |
|-----------------------------|------------------------------------------|
| `LoadLicelFile(path)`       | Load a single Licel file from disk       |
| `LoadLicelFileFromReader(r)`| Load from a binary stream                |
| `NewLicelPack(mask)`        | Load all files matching a glob pattern   |
| `NewLicelPackFromZip(path)` | Load all valid files from a ZIP archive  |

## License

LGPL v3 or later. See [LICENSE](LICENSE) for details.

## Related projects

This is a Python port of the Go package
[`github.com/physicist2018/licelfile`](https://github.com/physicist2018/licelfile).
Both packages produce bit‑identical results on the same input files.
