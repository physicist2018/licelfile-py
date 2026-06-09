"""
LicelPack — collection of Licel measurements loaded by file mask or from ZIP.

Provides loading and querying functionality for multiple Licel files.
"""

import glob
import io
import re
import zipfile
from datetime import datetime
from typing import Callable, ClassVar, Dict, List, Optional

import numpy as np

from .licelfile import (
    LicelFile,
    LicelProfilesList,
    LoadLicelFile,
    LoadLicelFileFromReader,
)
from .licelprofile import LicelProfile

_NPZ_PACK_VERSION: int = 1
"""Version identifier for .npz format."""


def _dt2ts(dt: Optional[datetime]) -> float:
    """Convert datetime to Unix timestamp (seconds since epoch). Returns NaN if None."""
    if dt is None:
        return float("nan")
    return dt.timestamp()


def _ts2dt(ts: float) -> Optional[datetime]:
    """Convert Unix timestamp to datetime. Returns None if NaN."""
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return None
    return datetime.fromtimestamp(ts)


def _is_valid_filename(filename: str) -> bool:
    """Check if filename matches the pattern 'b*.*'."""
    return bool(re.match(r"^[a-z].*\..+", filename))


def _update_time_range(pack: LicelPack, t: Optional[datetime]) -> None:
    """Update pack.StartTime and pack.StopTime based on measurement time t."""
    if t is None:
        return
    if pack.StartTime is None or t < pack.StartTime:
        pack.StartTime = t
    if pack.StopTime is None or t > pack.StopTime:
        pack.StopTime = t


class LicelPack:
    """Collection of Licel measurements."""

    __slots__ = ("StartTime", "StopTime", "Data")

    def __init__(self):
        self.StartTime: Optional[datetime] = None
        self.StopTime: Optional[datetime] = None
        self.Data: Dict[str, LicelFile] = {}

    def select_certain_wavelength(
        self, is_photon: bool, wavelength: float
    ) -> LicelProfilesList:
        """Select profiles by wavelength and type from all files in the pack."""
        result: LicelProfilesList = []
        for licf in self.Data.values():
            profile = licf.select_certain_wavelength(is_photon, wavelength)
            if profile.Wavelength != 0:
                result.append(profile)
        return result

    def truncate(self, rmax: float) -> None:
        """Truncate all profiles in all files to a maximum range."""
        for licf in self.Data.values():
            licf.truncate(rmax)

    def subtract_background(
        self,
        method: str = "mean",
        bgrRange: float = None,
        dark_file: "LicelFile" = None,
    ) -> None:
        """Subtract background from all profiles in all files of the pack.

        Args:
            method: One of "mean", "median", or "dark".
            bgrRange: Range in meters beyond which background is estimated
                      (used for "mean" and "median").
            dark_file: A single LicelFile with dark signal channels
                       (used for "dark"). Applied to every file in the pack.
        """
        for licf in self.Data.values():
            licf.subtract_background(
                method=method, bgrRange=bgrRange, dark_file=dark_file
            )

    def glue(
        self,
        wavelength: float,
        polarization: str,
        h1: float,
        h2: float,
    ) -> "LicelPack":
        """Glue analog and photon channels in all files of the pack (in-place).

        For each file that has both a photon and an analog channel with
        the given wavelength and polarization, creates (or updates) a
        glued profile. Files that don't have a matching pair are removed
        from the pack. StartTime and StopTime are recomputed from the
        remaining files.

        Returns:
            self, for chaining.
        """
        failed_names: list[str] = []
        for name, licf in self.Data.items():
            try:
                licf.glue(wavelength, polarization, h1, h2)
            except ValueError:
                failed_names.append(name)

        for name in failed_names:
            del self.Data[name]

        # Recompute StartTime/StopTime
        self.StartTime = None
        self.StopTime = None
        for licf in self.Data.values():
            _update_time_range(self, licf.MeasurementStartTime)

        return self

    def filter(self, f: "Callable[[LicelProfile], bool]") -> LicelProfilesList:
        """Collect profiles that satisfy a predicate across all files.

        Args:
            f: A callable that takes a LicelProfile and returns True
               to include it in the result.

        Returns:
            A flat list of LicelProfile objects for which the predicate
            returned True.
        """
        result: LicelProfilesList = []
        for licf in self.Data.values():
            result.extend(p for p in licf.Profiles if f(p))
        return result

    def filter_files(self, f: "Callable[[LicelFile], bool]") -> "LicelPack":
        """Filter LicelFile entries using a predicate function.

        Args:
            f: A callable that takes a LicelFile and returns True
               to include it in the result.

        Returns:
            A new LicelPack containing only the files for which
            the predicate returned True. The StartTime and StopTime
            are recomputed from the filtered set.
        """
        result = LicelPack()
        for name, licf in self.Data.items():
            if f(licf):
                result.Data[name] = licf
                _update_time_range(result, licf.MeasurementStartTime)
        return result

    def average(self) -> "LicelFile":
        """Create a new LicelFile by averaging all files in the pack.

        For each channel (profile index), computes the element-wise arithmetic
        mean of the corresponding profiles across all files in the pack.
        Metadata (site, coordinates, laser parameters, etc.) is taken from
        the first file in the pack.

        Returns:
            A new LicelFile with averaged profile data.

        Raises:
            ValueError: If the pack is empty or files have a different
                        number of profiles (NDatasets).
        """
        if not self.Data:
            raise ValueError("Cannot average an empty LicelPack")

        # Validate that all files have the same number of profiles
        nprofiles = None
        for name, licf in self.Data.items():
            if nprofiles is None:
                nprofiles = licf.NDatasets
            elif licf.NDatasets != nprofiles:
                raise ValueError(
                    f"File {name!r} has {licf.NDatasets} profiles, expected {nprofiles}"
                )

        # Take first file as template
        first_name = next(iter(self.Data))
        first_licf = self.Data[first_name]

        result = LicelFile()
        result.MeasurementSite = first_licf.MeasurementSite
        result.MeasurementStartTime = self.StartTime or first_licf.MeasurementStartTime
        result.MeasurementStopTime = self.StopTime or first_licf.MeasurementStopTime
        result.AltitudeAboveSeaLevel = first_licf.AltitudeAboveSeaLevel
        result.Longitude = first_licf.Longitude
        result.Latitude = first_licf.Latitude
        result.Zenith = first_licf.Zenith
        result.Laser1NShots = first_licf.Laser1NShots
        result.Laser1Freq = first_licf.Laser1Freq
        result.Laser2NShots = first_licf.Laser2NShots
        result.Laser2Freq = first_licf.Laser2Freq
        result.Laser3NShots = first_licf.Laser3NShots
        result.Laser3Freq = first_licf.Laser3Freq
        result.NDatasets = nprofiles

        # Averages profiles element-wise
        files_list = list(self.Data.values())
        for i in range(nprofiles):
            # Collect data arrays and find min NDataPoints
            data_arrays = []
            min_npts = None
            for licf in files_list:
                p = licf.Profiles[i]
                data_arrays.append(np.array(p.Data, dtype=np.float64))
                if min_npts is None or p.NDataPoints < min_npts:
                    min_npts = p.NDataPoints

            # Truncate all to minimum length and average
            averaged_data = np.mean([arr[:min_npts] for arr in data_arrays], axis=0)

            # Copy metadata from first file's profile
            template = first_licf.Profiles[i]
            avg_profile = LicelProfile()
            avg_profile.Active = template.Active
            avg_profile.Photon = template.Photon
            avg_profile.LaserType = template.LaserType
            avg_profile.NDataPoints = min_npts
            avg_profile.Reserved = template.Reserved[:]
            avg_profile.HighVoltage = template.HighVoltage
            avg_profile.BinWidth = template.BinWidth
            avg_profile.Wavelength = template.Wavelength
            avg_profile.Polarization = template.Polarization
            avg_profile.BinShift = template.BinShift
            avg_profile.DecBinShift = template.DecBinShift
            avg_profile.AdcBits = template.AdcBits
            avg_profile.NShots = template.NShots
            avg_profile.DiscrLevel = template.DiscrLevel
            avg_profile.DeviceID = template.DeviceID
            avg_profile.NCrate = template.NCrate
            avg_profile.Data = averaged_data.tolist()

            result.Profiles.append(avg_profile)

        result.FileLoaded = True
        return result

    def save(self) -> None:
        """Save all files in the pack to disk."""
        for fname, licf in self.Data.items():
            licf.save(fname)

    def to_npz(self, path: str) -> None:
        """Save the pack to a compressed NumPy .npz archive.

        The archive contains structured arrays with metadata and a 2D
        data matrix (NaN-padded to the longest profile).

        Args:
            path: Output file path (e.g. "pack.npz").
        """
        if not self.Data:
            raise ValueError("Cannot save an empty LicelPack to .npz")

        # Collect all profiles with file index
        file_names = list(self.Data.keys())
        profiles: list[tuple[int, LicelProfile]] = []
        for fi, name in enumerate(file_names):
            for p in self.Data[name].Profiles:
                profiles.append((fi, p))

        n_profiles = len(profiles)
        if n_profiles == 0:
            raise ValueError("No profiles in the pack")

        # Find max NDataPoints for padding
        max_npts = max(p.NDataPoints for _, p in profiles)

        # Build structured arrays
        file_meta_dtype = np.dtype(
            [
                ("name", "U256"),
                ("site", "U128"),
                ("start_time", "f8"),
                ("stop_time", "f8"),
                ("altitude", "f8"),
                ("longitude", "f8"),
                ("latitude", "f8"),
                ("zenith", "f8"),
                ("laser1_nshots", "i4"),
                ("laser1_freq", "i4"),
                ("laser2_nshots", "i4"),
                ("laser2_freq", "i4"),
                ("laser3_nshots", "i4"),
                ("laser3_freq", "i4"),
                ("ndatasets", "i4"),
            ]
        )

        n_files = len(file_names)
        file_meta = np.empty(n_files, dtype=file_meta_dtype)
        for fi, name in enumerate(file_names):
            lf = self.Data[name]
            file_meta[fi] = (
                name,
                lf.MeasurementSite,
                _dt2ts(lf.MeasurementStartTime),
                _dt2ts(lf.MeasurementStopTime),
                lf.AltitudeAboveSeaLevel,
                lf.Longitude,
                lf.Latitude,
                lf.Zenith,
                lf.Laser1NShots,
                lf.Laser1Freq,
                lf.Laser2NShots,
                lf.Laser2Freq,
                lf.Laser3NShots,
                lf.Laser3Freq,
                lf.NDatasets,
            )

        prof_meta_dtype = np.dtype(
            [
                ("file_index", "i4"),
                ("active", "?"),
                ("photon", "?"),
                ("laser_type", "i4"),
                ("npoints", "i4"),
                ("reserved", "3i4"),
                ("high_voltage", "i4"),
                ("bin_width", "f8"),
                ("wavelength", "f8"),
                ("polarization", "U8"),
                ("bin_shift", "i4"),
                ("dec_bin_shift", "i4"),
                ("adc_bits", "i4"),
                ("nshots", "i4"),
                ("discr_level", "f8"),
                ("device_id", "U4"),
                ("n_crate", "i4"),
            ]
        )

        prof_meta = np.empty(n_profiles, dtype=prof_meta_dtype)
        data = np.full((n_profiles, max_npts), np.nan, dtype=np.float64)
        for pi, (fi, p) in enumerate(profiles):
            prof_meta[pi] = (
                fi,
                p.Active,
                p.Photon,
                p.LaserType,
                p.NDataPoints,
                p.Reserved,
                p.HighVoltage,
                p.BinWidth,
                p.Wavelength,
                p.Polarization,
                p.BinShift,
                p.DecBinShift,
                p.AdcBits,
                p.NShots,
                p.DiscrLevel,
                p.DeviceID,
                p.NCrate,
            )
            data[pi, : p.NDataPoints] = p.Data[: p.NDataPoints]

        pack_start = _dt2ts(self.StartTime)
        pack_stop = _dt2ts(self.StopTime)

        np.savez_compressed(
            path,
            _pack_version=_NPZ_PACK_VERSION,
            _pack_start=pack_start,
            _pack_stop=pack_stop,
            file_meta=file_meta,
            prof_meta=prof_meta,
            data=data,
        )

    @classmethod
    def from_npz(cls, path: str) -> "LicelPack":
        """Load a LicelPack from a .npz archive created by to_npz().

        Args:
            path: Path to the .npz file.

        Returns:
            A new LicelPack instance.
        """
        data_npz = np.load(path)

        version = int(data_npz["_pack_version"])
        if version != _NPZ_PACK_VERSION:
            raise ValueError(
                f"Unsupported .npz version {version}, expected {_NPZ_PACK_VERSION}"
            )

        pack = cls()
        pack.StartTime = _ts2dt(float(data_npz["_pack_start"]))
        pack.StopTime = _ts2dt(float(data_npz["_pack_stop"]))

        file_meta = data_npz["file_meta"]
        prof_meta = data_npz["prof_meta"]
        data = data_npz["data"]

        # Reconstruct LicelFiles
        n_files = len(file_meta)
        for fi in range(n_files):
            fm = file_meta[fi]
            lf = LicelFile()
            lf.MeasurementSite = str(fm["site"])
            lf.MeasurementStartTime = _ts2dt(float(fm["start_time"]))
            lf.MeasurementStopTime = _ts2dt(float(fm["stop_time"]))
            lf.AltitudeAboveSeaLevel = float(fm["altitude"])
            lf.Longitude = float(fm["longitude"])
            lf.Latitude = float(fm["latitude"])
            lf.Zenith = float(fm["zenith"])
            lf.Laser1NShots = int(fm["laser1_nshots"])
            lf.Laser1Freq = int(fm["laser1_freq"])
            lf.Laser2NShots = int(fm["laser2_nshots"])
            lf.Laser2Freq = int(fm["laser2_freq"])
            lf.Laser3NShots = int(fm["laser3_nshots"])
            lf.Laser3Freq = int(fm["laser3_freq"])
            lf.NDatasets = int(fm["ndatasets"])
            lf.FileLoaded = True

            # Find profiles belonging to this file
            file_prof_indices = np.where(prof_meta["file_index"] == fi)[0]
            for pi in file_prof_indices:
                pm = prof_meta[pi]
                p = LicelProfile()
                p.Active = bool(pm["active"])
                p.Photon = bool(pm["photon"])
                p.LaserType = int(pm["laser_type"])
                p.NDataPoints = int(pm["npoints"])
                p.Reserved = list(pm["reserved"])
                p.HighVoltage = int(pm["high_voltage"])
                p.BinWidth = float(pm["bin_width"])
                p.Wavelength = float(pm["wavelength"])
                p.Polarization = str(pm["polarization"])
                p.BinShift = int(pm["bin_shift"])
                p.DecBinShift = int(pm["dec_bin_shift"])
                p.AdcBits = int(pm["adc_bits"])
                p.NShots = int(pm["nshots"])
                p.DiscrLevel = float(pm["discr_level"])
                p.DeviceID = str(pm["device_id"])
                p.NCrate = int(pm["n_crate"])
                npts = int(pm["npoints"])
                p.Data = data[pi, :npts].tolist()
                lf.Profiles.append(p)

            pack.Data[str(fm["name"])] = lf

        data_npz.close()
        return pack

    def save_to_zip(
        self,
        zip_path: str,
        compression: int = zipfile.ZIP_DEFLATED,
        compresslevel: int = 6,
    ) -> None:
        """Save all files in the pack to a ZIP archive.

        Args:
            zip_path: Path to the output ZIP file.
            compression: Compression method (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, etc.).
            compresslevel: Compression level (0-9). Default 6.
                          - 0: no compression (store only)
                          - 1: fastest, least compression
                          - 6: default balance
                          - 9: slowest, most compression

        Raises:
            ValueError: If compresslevel is outside the valid range for the chosen method.
        """
        with zipfile.ZipFile(
            zip_path, "w", compression=compression, compresslevel=compresslevel
        ) as zw:
            for fname, licf in self.Data.items():
                # Strip leading slash if present (used in NewLicelPackFromZip)
                arcname = fname.lstrip("/")
                zw.writestr(arcname, licf.to_bytes(arcname))

    def __repr__(self) -> str:
        return (
            f"LicelPack(start={self.StartTime}, stop={self.StopTime}, "
            f"files={len(self.Data)})"
        )

    def to_dict(self) -> dict:
        """Convert LicelPack to a dictionary (for JSON serialization)."""
        return {
            "start_time": (self.StartTime.isoformat() if self.StartTime else None),
            "stop_time": (self.StopTime.isoformat() if self.StopTime else None),
            "data": {name: lf.to_dict() for name, lf in self.Data.items()},
        }


def NewLicelPack(mask: str) -> LicelPack:
    """Load files according to a glob mask."""
    pack = LicelPack()
    files = glob.glob(mask)
    if not files:
        raise FileNotFoundError(f"No files found matching mask: {mask}")

    for fname in files:
        licf = LoadLicelFile(fname)
        pack.Data[fname] = licf

        t = licf.MeasurementStartTime
        _update_time_range(pack, t)

    return pack


def NewLicelPackFromZip(zip_path: str) -> LicelPack:
    """Load files from a ZIP archive."""
    pack = LicelPack()

    with zipfile.ZipFile(zip_path, "r") as zr:
        for info in zr.infolist():
            fname = info.filename
            if not _is_valid_filename(fname):
                continue

            # Read file contents into memory
            file_bytes = zr.read(fname)

            # Load LicelFile from bytes
            licf = LoadLicelFileFromReader(io.BytesIO(file_bytes), len(file_bytes))

            full_path = "/" + fname
            pack.Data[full_path] = licf

            t = licf.MeasurementStartTime
            _update_time_range(pack, t)

    return pack
