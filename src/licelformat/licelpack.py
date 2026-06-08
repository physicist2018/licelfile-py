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

from .licelfile import (
    LicelFile,
    LicelProfilesList,
    LoadLicelFile,
    LoadLicelFileFromReader,
)


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

    def filter(self, f: "Callable[[LicelFile], bool]") -> "LicelPack":
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
