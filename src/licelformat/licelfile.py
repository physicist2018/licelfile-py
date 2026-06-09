"""
LicelFile — structure representing a single Licel measurement.

Provides loading, saving, and querying functionality for Licel format files.
"""

from datetime import datetime
from io import BufferedReader, BytesIO
from typing import IO, Callable, List, Optional

import numpy as np

from .licelprofile import LicelProfile, _str2float, _str2int

LicelProfilesList = List[LicelProfile]


class LicelFile:
    """Structure representing a single Licel measurement."""

    __slots__ = (
        "MeasurementSite",
        "MeasurementStartTime",
        "MeasurementStopTime",
        "AltitudeAboveSeaLevel",
        "Longitude",
        "Latitude",
        "Zenith",
        "Laser1NShots",
        "Laser1Freq",
        "Laser2NShots",
        "Laser2Freq",
        "NDatasets",
        "Laser3NShots",
        "Laser3Freq",
        "FileLoaded",
        "Profiles",
    )

    def __init__(self):
        self.MeasurementSite: str = ""
        self.MeasurementStartTime: Optional[datetime] = None
        self.MeasurementStopTime: Optional[datetime] = None
        self.AltitudeAboveSeaLevel: float = 0.0
        self.Longitude: float = 0.0
        self.Latitude: float = 0.0
        self.Zenith: float = 0.0
        self.Laser1NShots: int = 0
        self.Laser1Freq: int = 0
        self.Laser2NShots: int = 0
        self.Laser2Freq: int = 0
        self.NDatasets: int = 0
        self.Laser3NShots: int = 0
        self.Laser3Freq: int = 0
        self.FileLoaded: bool = False
        self.Profiles: LicelProfilesList = []

    def select_certain_wavelength(
        self, is_photon: bool, wavelength: float
    ) -> LicelProfile:
        """Select a profile by its wavelength and type."""
        for profile in self.Profiles:
            if profile.Photon == is_photon and profile.Wavelength == wavelength:
                return profile
        return LicelProfile()

    def truncate(self, rmax: float) -> None:
        """Truncate all profiles to a maximum range."""
        for p in self.Profiles:
            p.truncate(rmax)

    def glue(
        self,
        wavelength: float,
        polarization: str,
        h1: float,
        h2: float,
    ) -> LicelProfile:
        """Glue (merge) photon and analog channels of the same wavelength.

        Finds a photon channel (isPhoton) and an analog channel (isAnalog)
        with the given wavelength and polarization, computes the mean
        ratio k = analog / photon in the altitude range [h1, h2], and
        produces a merged ('glued') profile:

          h < h1   : analog data
          h1..h2   : (analog + photon * k) / 2
          h > h2   : photon * k

        Args:
            wavelength: Laser wavelength in nm.
            polarization: Polarization string ("o", "s", or "").
            h1: Start of the merge interval (meters).
            h2: End of the merge interval (meters).

        Returns:
            The newly created glued LicelProfile (DeviceID == 'BG').

        Raises:
            ValueError: If no matching photon/analog pair is found or
                        BinWidth is zero.
        """
        p1 = None  # photon
        p2 = None  # analog
        for p in self.Profiles:
            if p.Wavelength != wavelength or p.Polarization != polarization:
                continue
            if p.isPhoton:
                p1 = p
            elif p.isAnalog:
                p2 = p

        if p1 is None or p2 is None:
            raise ValueError(
                f"No matching photon/analog pair for wavelength={wavelength}, "
                f"polarization={polarization!r}"
            )

        if p1.BinWidth <= 0 or p2.BinWidth <= 0:
            raise ValueError("BinWidth must be positive")

        npts = min(p1.NDataPoints, p2.NDataPoints)
        bw = p1.BinWidth
        n1 = min(int(h1 / bw), npts)
        n2 = min(int(h2 / bw), npts)

        if n2 <= n1:
            raise ValueError(f"h2 ({h2}) must be greater than h1 ({h1})")

        # Convert to numpy arrays for arithmetic operations
        p1_data = np.array(p1.Data, dtype=np.float64)
        p2_data = np.array(p2.Data, dtype=np.float64)

        # Compute mean ratio k = analog / photon in the overlap zone
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = p2_data[n1:n2] / p1_data[n1:n2]
            ratio = ratio[np.isfinite(ratio)]
        if len(ratio) == 0:
            raise ValueError("No valid ratio values in the h1..h2 interval")
        k = float(np.mean(ratio))

        # Build the glued data
        glued_data_arr = np.empty(npts, dtype=np.float64)
        # h < h1: analog
        glued_data_arr[:n1] = p2_data[:n1]
        # h1..h2: (analog + photon * k) / 2
        glued_data_arr[n1:n2] = (p2_data[n1:n2] + p1_data[n1:n2] * k) * 0.5
        # h > h2: photon * k
        glued_data_arr[n2:] = p1_data[n2:npts] * k

        # Check if a glued profile for this wavelength/polarization already exists
        existing = None
        for p in self.Profiles:
            if (
                p.isGlued
                and p.Wavelength == wavelength
                and p.Polarization == polarization
            ):
                existing = p
                break

        if existing is not None:
            # Update in-place
            glued = existing
            glued.NDataPoints = npts
            glued.Data = glued_data_arr.tolist()
        else:
            # Create a new profile
            glued = LicelProfile()
            # Copy metadata from analog channel
            glued.Active = p2.Active
            glued.Photon = False
            glued.LaserType = p2.LaserType
            glued.NDataPoints = npts
            glued.Reserved = p2.Reserved[:]
            glued.HighVoltage = p2.HighVoltage
            glued.BinWidth = p2.BinWidth
            glued.Wavelength = p2.Wavelength
            glued.Polarization = p2.Polarization
            glued.BinShift = p2.BinShift
            glued.DecBinShift = p2.DecBinShift
            glued.AdcBits = p2.AdcBits
            glued.NShots = p2.NShots
            glued.DiscrLevel = p2.DiscrLevel
            glued.DeviceID = "BG"
            glued.NCrate = p2.NCrate
            glued.Data = glued_data_arr.tolist()

            self.Profiles.append(glued)
            self.NDatasets += 1

        return glued

    def subtract_background(
        self,
        method: str = "mean",
        bgrRange: float = None,
        dark_file: "LicelFile" = None,
    ) -> None:
        """Subtract background from all profiles in the file.

        Args:
            method: One of "mean", "median", or "dark".
            bgrRange: Range in meters beyond which background is estimated
                      (used for "mean" and "median").
            dark_file: A LicelFile containing dark signal profiles
                       (used for "dark"). For each profile, the matching
                       dark profile is found by wavelength, polarization,
                       and channel type.

        Raises:
            ValueError: If method is invalid or required arguments are missing.
        """
        if method == "dark":
            if dark_file is None:
                raise ValueError("dark_file is required for method='dark'")
            for p in self.Profiles:
                dark_p = None
                for dp in dark_file.Profiles:
                    if (
                        dp.Wavelength == p.Wavelength
                        and dp.Polarization == p.Polarization
                        and dp.isPhoton == p.isPhoton
                        and dp.isAnalog == p.isAnalog
                    ):
                        dark_p = dp
                        break
                if dark_p is None:
                    raise ValueError(
                        f"No matching dark profile for wavelength={p.Wavelength}, "
                        f"polarization={p.Polarization!r}, "
                        f"isPhoton={p.isPhoton}, isAnalog={p.isAnalog}"
                    )
                p.subtract_background(method="dark", dark_profile=dark_p)
        else:
            for p in self.Profiles:
                p.subtract_background(method=method, bgrRange=bgrRange)

    def filter(self, f: "Callable[[LicelProfile], bool]") -> LicelProfilesList:
        """Filter profiles using a predicate function.

        Args:
            f: A callable that takes a LicelProfile and returns True
               to include it in the result.

        Returns:
            List of profiles for which the predicate returned True.
        """
        return [p for p in self.Profiles if f(p)]

    def save(self, fname: str) -> None:
        """Save the Licel file to disk."""
        with open(fname, "wb") as f:
            _ = f.write(self.to_bytes(fname))

    def to_bytes(self, fname: str) -> bytes:
        """Serialize the LicelFile to bytes (as stored in a .lic file).

        Args:
            fname: Filename to embed in the first header line.

        Returns:
            Complete file content as bytes.
        """
        buf = BytesIO()
        buf.write(self._format_first_line(fname).encode("latin-1"))
        buf.write(self._format_second_line().encode("latin-1"))
        buf.write(self._format_third_line().encode("latin-1"))
        for profile in self.Profiles:
            buf.write(profile.metadata().encode("latin-1"))
        buf.write(b"\r\n")
        for profile in self.Profiles:
            buf.write(profile.profile())
        return buf.getvalue()

    def _format_first_line(self, fname: str) -> str:
        """Return the first line of a LICEL file."""
        return f" {fname:<77s}\r\n"

    def _format_second_line(self) -> str:
        """Return the second line of a LICEL file."""
        s = (
            f" {self.MeasurementSite} "
            f"{self.MeasurementStartTime.strftime('%d/%m/%Y')} "
            f"{self.MeasurementStartTime.strftime('%H:%M:%S')} "
            f"{self.MeasurementStopTime.strftime('%d/%m/%Y')} "
            f"{self.MeasurementStopTime.strftime('%H:%M:%S')} "
            f"{self.AltitudeAboveSeaLevel:04.0f} "
            f"{self.Longitude:06.1f} "
            f"{self.Latitude:06.1f} "
            f"{self.Zenith:02.0f}"
        )
        return f"{s:<78s}\r\n"

    def _format_third_line(self) -> str:
        """Return the third line of a LICEL file."""
        s = (
            f" {self.Laser1NShots:07d} {self.Laser1Freq:04d} "
            f"{self.Laser2NShots:07d} {self.Laser2Freq:04d} "
            f"{self.NDatasets:02d} {self.Laser3NShots:07d} "
            f"{self.Laser3Freq:04d}"
        )
        return f"{s:<78s}\r\n"

    def to_dict(self) -> dict:
        """Convert LicelFile to a dictionary (for JSON serialization)."""
        return {
            "location": self.MeasurementSite,
            "start_time": (
                self.MeasurementStartTime.isoformat()
                if self.MeasurementStartTime
                else None
            ),
            "stop_time": (
                self.MeasurementStopTime.isoformat()
                if self.MeasurementStopTime
                else None
            ),
            "lidar_altitude": self.AltitudeAboveSeaLevel,
            "longitude": self.Longitude,
            "latitude": self.Latitude,
            "zenith": self.Zenith,
            "laser1_nshots": self.Laser1NShots,
            "laser1_freq": self.Laser1Freq,
            "laser2_nshots": self.Laser2NShots,
            "laser2_freq": self.Laser2Freq,
            "dataset_count": self.NDatasets,
            "laser3_nshots": self.Laser3NShots,
            "laser3_freq": self.Laser3Freq,
            "datasets": [p.to_dict() for p in self.Profiles],
        }

    def __repr__(self) -> str:
        return (
            f"LicelFile(site={self.MeasurementSite}, "
            f"start={self.MeasurementStartTime}, "
            f"profiles={len(self.Profiles)})"
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _parse_time(s: str) -> datetime:
    """Parse datetime string in format 'dd/mm/yyyy hh:mm:ss'."""
    return datetime.strptime(s, "%d/%m/%Y %H:%M:%S")


def _read_and_trim_line(r: BufferedReader) -> str:
    """Read a line from reader and trim right-side whitespace."""
    line = r.readline()
    if not line:
        raise EOFError("Unexpected end of file while reading header line")
    return line.decode("latin-1").rstrip("\t\r\n ")


def _skip_crlf(r: BufferedReader) -> None:
    """Skip CR+LF (2 bytes) from reader."""
    crlf = r.read(2)
    if len(crlf) < 2:
        raise EOFError("Unexpected end of file while skipping CRLF")


def _bytes_to_float64_array(b: bytes) -> np.ndarray:
    """Convert raw bytes to numpy float64 array (little-endian int32 → float64)."""
    # Interpret bytes as little-endian int32, then cast to float64
    arr = np.frombuffer(b, dtype=np.int32).astype(np.float64)
    return arr


# ---------------------------------------------------------------------------
# Main loading functions
# ---------------------------------------------------------------------------


def LoadLicelFile(fname: str) -> LicelFile:
    """Load a LicelFile from the specified file path."""
    with open(fname, "rb") as f:
        return _load_licel_file_from_buffered_reader(f)


def LoadLicelFileFromReader(stream: IO[bytes], size: int = 0) -> LicelFile:
    """Load a LicelFile from a binary reader/stream."""
    # Wrap in BufferedReader if needed
    if isinstance(stream, BufferedReader):
        r = stream
    else:
        r = BufferedReader(stream)  # type: ignore[arg-type]
    return _load_licel_file_from_buffered_reader(r)


def _load_licel_file_from_buffered_reader(r: BufferedReader) -> LicelFile:
    """Core loading logic from a buffered binary reader."""
    licf = LicelFile()

    # Skip first line (contains filename or is empty)
    _read_and_trim_line(r)

    # Second line: basic information
    header = _read_and_trim_line(r)
    tmp = header.split()

    licf.MeasurementSite = tmp[0]
    licf.MeasurementStartTime = _parse_time(tmp[1] + " " + tmp[2])
    licf.MeasurementStopTime = _parse_time(tmp[3] + " " + tmp[4])
    licf.AltitudeAboveSeaLevel = _str2float(tmp[5])
    licf.Longitude = _str2float(tmp[6])
    licf.Latitude = _str2float(tmp[7])
    licf.Zenith = _str2float(tmp[8])

    # Third line: laser parameters
    header = _read_and_trim_line(r)
    tmp = header.split()
    licf.Laser1NShots = _str2int(tmp[0])
    licf.Laser1Freq = _str2int(tmp[1])
    licf.Laser2NShots = _str2int(tmp[2])
    licf.Laser2Freq = _str2int(tmp[3])
    licf.NDatasets = _str2int(tmp[4])
    licf.Laser3NShots = _str2int(tmp[5])
    licf.Laser3Freq = _str2int(tmp[6])

    # Profiles (headers)
    licf.Profiles = []
    for _ in range(licf.NDatasets):
        header = _read_and_trim_line(r)
        licf.Profiles.append(LicelProfile(header))

    # After headers — binary data
    _skip_crlf(r)

    for i in range(licf.NDatasets):
        n_bytes = licf.Profiles[i].NDataPoints * 4
        pr_tmp = r.read(n_bytes)
        if len(pr_tmp) < n_bytes:
            raise EOFError("Error reading binary data: unexpected end of file")

        licf.Profiles[i].Data = _bytes_to_float64_array(pr_tmp)

        # Apply scaling using the profile's own scale_factor
        licf.Profiles[i].Data *= licf.Profiles[i].scale_factor()

        _skip_crlf(r)

    licf.FileLoaded = True
    return licf
