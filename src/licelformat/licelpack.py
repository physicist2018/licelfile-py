"""
LicelPack — collection of Licel measurements loaded by file mask or from ZIP.

Provides loading and querying functionality for multiple Licel files.
"""

import glob
import io
import re
import zipfile
from datetime import datetime
from typing import Callable, Dict, Optional

import numpy as np

from .licelfile import (
    LicelFile,
    LicelProfilesList,
    LoadLicelFile,
    LoadLicelFileFromReader,
)
from .licelprofile import LicelProfile


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
        """Glue analog and photon channels, returning files that succeeded.

        For each file that has both a photon and an analog channel with
        the given wavelength and polarization, creates a glued profile.

        Returns:
            A new LicelPack containing only the files where glue succeeded.
        """
        result = LicelPack()
        for name, licf in self.Data.items():
            try:
                licf.glue(wavelength, polarization, h1, h2)
                result.Data[name] = licf
                # _update_time_range(result, licf.MeasurementStartTime)
            except ValueError:
                pass
        return result

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
