# licelformat API Reference

## Module‑level functions

| Function | Description |
|---|---|
| `LoadLicelFile(path: str) -> LicelFile` | Load a single Licel file from disk |
| `LoadLicelFileFromReader(stream, size=0) -> LicelFile` | Load from a binary stream |
| `NewLicelPack(mask: str) -> LicelPack` | Load all files matching a glob pattern |
| `NewLicelPackFromZip(path: str) -> LicelPack` | Load all valid files from a ZIP archive |
| `to_netcdf(pack: LicelPack, path: str, format: str = "NETCDF4") -> None` | Save a `LicelPack` to NetCDF (CF-1.8). `format` can be `"NETCDF4"` (default), `"NETCDF3_CLASSIC"`, or `"NETCDF3_64BIT"`. Requires `netCDF4`. |
| `from_netcdf(path: str) -> LicelPack` | Load a `LicelPack` from a NetCDF file. Requires `netCDF4`. |

---

## `LicelProfile`

Represents a single measurement channel (profile) in a Licel file.

### Fields

| Field | Type | Description |
|---|---|---|
| `Active` | `bool` | Channel active flag |
| `Photon` | `bool` | `True` for photon counting |
| `LaserType` | `int` | Laser index (1, 2, 3) |
| `NDataPoints` | `int` | Number of bins |
| `Reserved` | `list[int]` | 3 reserved values `[r0, r1, r2]` |
| `HighVoltage` | `int` | PMT high voltage (V) |
| `BinWidth` | `float` | Range bin width (m) |
| `Wavelength` | `float` | Laser wavelength (nm) |
| `Polarization` | `str` | `"o"`, `"s"`, or `"p"` |
| `BinShift` | `int` | Bin shift |
| `DecBinShift` | `int` | Decimal bin shift |
| `AdcBits` | `int` | ADC resolution (bits) |
| `NShots` | `int` | Number of laser shots |
| `DiscrLevel` | `float` | Discriminator level (mV) |
| `DeviceID` | `str` | 2-char device identifier |
| `NCrate` | `int` | Crate slot number |
| `Data` | `list[float]` | Scaled float64 profile data (list, not ndarray) |

### Properties

| Property | Type | Description |
|---|---|---|
| `isPhoton` | `bool` | `True` if `DeviceID == 'BC' and Photon == True` |
| `isAnalog` | `bool` | `True` if `DeviceID == 'BT'` |
| `isGlued` | `bool` | `True` if `DeviceID == 'BG'` |

### Methods

#### `__init__(self, line: str = None)`

Parse a single metadata line and create a profile. If `line` is `None`, creates an empty profile.

#### `truncate(rmax: float) -> None`

Truncate profile data to a maximum range.

**Args:**
- `rmax` — Maximum range in meters. Points beyond this range are removed.

---

#### `subtract_background(method="mean", bgrRange=None, dark_profile=None) -> None`

Subtract background signal from the profile.

**Args:**
- `method` — One of `"mean"`, `"median"`, or `"dark"`.
- `bgrRange` — Range in meters beyond which background is estimated (required for `"mean"` and `"median"`).
- `dark_profile` — A `LicelProfile` with dark signal data (required for `"dark"`).

**Raises:**
- `ValueError` — If required arguments are missing, `BinWidth` is zero, `bgrRange` exceeds data length, or data lengths don't match.

**Behaviour by method:**
- `"mean"` — Computes `bg = mean(Data[n:])` where `n = bgrRange / BinWidth`, then subtracts `bg` from all points.
- `"median"` — Same but uses `median` instead of `mean`.
- `"dark"` — Subtracts `dark_profile.Data` point by point.

---

#### `metadata() -> str`

Return the 78-character metadata string (header line) for this profile.

---

#### `profile() -> bytes`

Convert profile data to binary bytes (little-endian int32), applying inverse scaling.

---

#### `scale_factor() -> float`

Return the scale factor used during loading:

- Analog: `DiscrLevel × 1000 / (2^AdcBits × NShots)` → mV
- Photon: `1 / (NShots × 0.05)` → MHz

---

#### `to_dict() -> dict`

Convert profile to a dictionary (for JSON serialization).

---

## `LicelFile`

Structure representing a single Licel measurement file.

### Fields

| Field | Type | Description |
|---|---|---|
| `MeasurementSite` | `str` | Site name |
| `MeasurementStartTime` | `Optional[datetime]` | Measurement start |
| `MeasurementStopTime` | `Optional[datetime]` | Measurement stop |
| `AltitudeAboveSeaLevel` | `float` | Lidar altitude (m) |
| `Longitude` | `float` | Longitude (deg) |
| `Latitude` | `float` | Latitude (deg) |
| `Zenith` | `float` | Zenith angle (deg) |
| `Laser1NShots` | `int` | Laser 1 shot count |
| `Laser1Freq` | `int` | Laser 1 frequency (Hz) |
| `Laser2NShots` | `int` | Laser 2 shot count |
| `Laser2Freq` | `int` | Laser 2 frequency (Hz) |
| `NDatasets` | `int` | Number of profiles |
| `Laser3NShots` | `int` | Laser 3 shot count |
| `Laser3Freq` | `int` | Laser 3 frequency (Hz) |
| `FileLoaded` | `bool` | `True` after successful load |
| `Profiles` | `list[LicelProfile]` | All profiles in the file |

### Methods

#### `select_certain_wavelength(is_photon: bool, wavelength: float) -> LicelProfile`

Find a profile by photon flag and wavelength.

**Returns:** The matching profile, or an empty `LicelProfile()` (with `Wavelength == 0`) if not found.

---

#### `truncate(rmax: float) -> None`

Truncate all profiles to a maximum range in meters. Delegates to `LicelProfile.truncate()`.

---

#### `glue(wavelength: float, polarization: str, h1: float, h2: float) -> LicelProfile`

Merge (glue) analog and photon channels of the same wavelength.

Finds a photon channel (`isPhoton == True`) and an analog channel (`isAnalog == True`) with matching `wavelength` and `polarization`. Computes the mean ratio `k = analog / photon` in the altitude range `[h1, h2]`, then builds a merged profile:

| Altitude | Data |
|---|---|
| `h < h1` | Raw analog data |
| `h1 ≤ h ≤ h2` | `(analog + photon × k) / 2` |
| `h > h2` | `photon × k` |

If a glued profile with the same `wavelength` and `polarization` already exists (i.e. `DeviceID == "BG"`), it is **updated in-place** instead of creating a new one. This means calling `glue()` twice with the same wavelength/polarization does not duplicate the profile — the data is overwritten and `NDatasets` is not incremented.

**Args:**
- `wavelength` — Laser wavelength in nm.
- `polarization` — Polarization string (`"o"`, `"s"`, or `""`).
- `h1` — Start of merge interval (meters).
- `h2` — End of merge interval (meters).

**Returns:** The glued `LicelProfile` (with `DeviceID == "BG"`).

**Raises:**
- `ValueError` — If no matching pair is found, `BinWidth` is zero, `h2 ≤ h1`, or no valid ratio values in the interval.

---

#### `subtract_background(method="mean", bgrRange=None, dark_file=None) -> None`

Subtract background from all profiles in the file.

**Args:**
- `method` — One of `"mean"`, `"median"`, or `"dark"`.
- `bgrRange` — Range in meters beyond which background is estimated (for `"mean"`/`"median"`).
- `dark_file` — A `LicelFile` containing dark signal channels (for `"dark"`).

**For `method="dark"`:** For each profile in the file, finds a matching dark profile in `dark_file` by `Wavelength`, `Polarization`, `isPhoton`, and `isAnalog`, and subtracts it point by point. Raises `ValueError` if no match is found.

---

#### `filter(f: Callable[[LicelProfile], bool]) -> list[LicelProfile]`

Filter profiles using a predicate function.

**Args:**
- `f` — A callable that takes a `LicelProfile` and returns `True` to include it.

**Returns:** List of profiles matching the predicate.

---

#### `save(fname: str) -> None`

Save the Licel file to disk as `fname + "1"`.

---

#### `to_bytes(fname: str) -> bytes`

Serialize the entire `LicelFile` to raw bytes (header + metadata + binary profile data, as stored in a `.lic` file).

**Args:**
- `fname` — Filename to embed in the first header line.

---

#### `to_dict() -> dict`

Convert the file to a dictionary (for JSON serialization).

---

## `LicelPack`

Collection of Licel measurements loaded by file mask or from ZIP archive.

### Fields

| Field | Type | Description |
|---|---|---|
| `StartTime` | `Optional[datetime]` | Earliest measurement start across all files |
| `StopTime` | `Optional[datetime]` | Latest measurement time across all files |
| `Data` | `dict[str, LicelFile]` | Filename → `LicelFile` mapping |

### Methods

#### `select_certain_wavelength(is_photon: bool, wavelength: float) -> list[LicelProfile]`

Collect profiles with the given wavelength and type from all files in the pack.

---

#### `truncate(rmax: float) -> None`

Truncate all profiles in all files to a maximum range in meters. Delegates to `LicelFile.truncate()`.

---

#### `subtract_background(method="mean", bgrRange=None, dark_file=None) -> None`

Subtract background from all profiles in all files of the pack.

**Args:**
- `method` — One of `"mean"`, `"median"`, or `"dark"`.
- `bgrRange` — Range in meters beyond which background is estimated (for `"mean"`/`"median"`).
- `dark_file` — A single `LicelFile` with dark signal channels, applied to every file in the pack.

---

#### `glue(wavelength: float, polarization: str, h1: float, h2: float) -> LicelPack`

Glue analog and photon channels in all files of the pack (in-place).

For each file that has both a photon and an analog channel with the given `wavelength` and `polarization`, creates (or updates) a glued profile. Files that don't have a matching pair are **removed** from the pack. `StartTime` and `StopTime` are recomputed from the remaining files.

**Returns:** `self` (for method chaining).

---

#### `filter(f: Callable[[LicelProfile], bool]) -> list[LicelProfile]`

Collect profiles that satisfy a predicate across all files.

**Args:**
- `f` — A callable that takes a `LicelProfile` and returns `True` to include it.

**Returns:** A flat list of matching `LicelProfile` objects.

---

#### `average() -> LicelFile`

Create a new `LicelFile` by averaging all files in the pack.

For each channel (profile index), computes the element-wise arithmetic mean of the corresponding profiles across all files in the pack. If profiles have different `NDataPoints`, the shortest length is used (others are truncated). Metadata (site, coordinates, laser parameters, etc.) is taken from the first file in the pack; `StartTime`/`StopTime` are taken from the pack's own fields.

**Raises:**
- `ValueError` — If the pack is empty or files have a different number of profiles (`NDatasets`).

---

#### `filter_files(f: Callable[[LicelFile], bool]) -> LicelPack`

Filter files using a predicate function.

**Args:**
- `f` — A callable that takes a `LicelFile` and returns `True` to include it.

**Returns:** A new `LicelPack` containing only the matching files. `StartTime` and `StopTime` are recomputed from the filtered set (shallow copy — shares the same `LicelFile` objects).

---

#### `save() -> None`

Save all files in the pack to disk. Delegates to `LicelFile.save()` for each file.

---

#### `to_npz(path: str) -> None`

Save the pack to a compressed NumPy `.npz` archive. Contains structured arrays with metadata and a 2D data matrix (NaN-padded to the longest profile).

**Args:**
- `path` — Output file path (e.g. `"pack.npz"`).

**Raises:**
- `ValueError` — If the pack is empty or has no profiles.

---

#### `from_npz(path: str) -> LicelPack` *(classmethod)*

Load a `LicelPack` from a `.npz` archive created by `to_npz()`.

**Args:**
- `path` — Path to the `.npz` file.

**Returns:** A new `LicelPack` instance.

---

#### `save_to_zip(zip_path: str, compression=ZIP_DEFLATED, compresslevel=6) -> None`

Save all files in the pack to a ZIP archive.

**Args:**
- `zip_path` — Path to the output ZIP file.
- `compression` — Compression method (`zipfile.ZIP_STORED`, `zipfile.ZIP_DEFLATED`, etc.).
- `compresslevel` — Compression level (0–9). Default 6. Only applies to methods that support it.

**Raises:**
- `ValueError` — If `compresslevel` is outside the valid range for the chosen method.

---

#### `to_dict() -> dict`

Convert the pack to a dictionary (for JSON serialization).
