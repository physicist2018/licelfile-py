"""
NetCDF I/O for LicelPack.

Provides save / load of LicelPack data in NetCDF format (CF-1.8 conventions).
Requires the optional ``netCDF4`` package.
"""

from datetime import datetime
from typing import Optional

import numpy as np

from .licelfile import LicelFile
from .licelpack import LicelPack
from .licelprofile import LicelProfile

__all__ = ["to_netcdf", "from_netcdf"]

_NC_VERSION: int = 1


def to_netcdf(pack: LicelPack, path: str) -> None:
    """Save a LicelPack to a NetCDF file.

    Args:
        pack: The LicelPack to save.
        path: Output file path (e.g. "measurements.nc").

    Raises:
        ImportError: If netCDF4 is not installed.
        ValueError: If the pack is empty or has no profiles.
    """
    if not pack.Data:
        raise ValueError("Cannot save an empty LicelPack to NetCDF")

    try:
        from netCDF4 import Dataset
    except ImportError:
        raise ImportError(
            "netCDF4 is required for NetCDF I/O. Install it with: pip install netCDF4"
        )

    # Collect all profiles with file index
    file_names = list(pack.Data.keys())
    profiles: list[tuple[int, LicelProfile]] = []
    for fi, name in enumerate(file_names):
        for p in pack.Data[name].Profiles:
            profiles.append((fi, p))

    n_profiles = len(profiles)
    if n_profiles == 0:
        raise ValueError("No profiles in the pack")

    # Find max NDataPoints for the range dimension
    max_npts = max(p.NDataPoints for _, p in profiles)
    bw = profiles[0][1].BinWidth  # use first profile's bin width for range

    with Dataset(path, "w", format="NETCDF4") as ds:
        # --- Global attributes ---
        ds.setncattr("Conventions", "CF-1.8")
        ds.setncattr("source", f"licelformat v{_NC_VERSION}")
        ds.setncattr("_licelformat_version", _NC_VERSION)

        # --- Dimensions ---
        ds.createDimension("file", n_files := len(file_names))
        ds.createDimension("profile", n_profiles)
        ds.createDimension("range", max_npts)

        # ---- Range variable ----
        range_var = ds.createVariable(
            "range",
            "f8",
            ("range",),
            fill_value=np.nan,
        )
        range_var.long_name = "range from lidar"
        range_var.units = "meters"
        range_var[:] = np.arange(max_npts, dtype=np.float64) * bw

        # ---- File-level variables ----
        file_name_var = ds.createVariable("file_name", str, ("file",))
        file_name_var.long_name = "original file name"

        site_var = ds.createVariable("site", str, ("file",))
        site_var.long_name = "measurement site"

        start_time_var = ds.createVariable(
            "start_time",
            "f8",
            ("file",),
            fill_value=np.nan,
        )
        start_time_var.long_name = "measurement start time"
        start_time_var.units = "seconds since 1970-01-01 00:00:00 UTC"
        start_time_var.calendar = "standard"

        stop_time_var = ds.createVariable(
            "stop_time",
            "f8",
            ("file",),
            fill_value=np.nan,
        )
        stop_time_var.long_name = "measurement stop time"
        stop_time_var.units = "seconds since 1970-01-01 00:00:00 UTC"
        stop_time_var.calendar = "standard"

        lon_var = ds.createVariable("longitude", "f8", ("file",), fill_value=np.nan)
        lon_var.long_name = "longitude"
        lon_var.units = "degrees_east"

        lat_var = ds.createVariable("latitude", "f8", ("file",), fill_value=np.nan)
        lat_var.long_name = "latitude"
        lat_var.units = "degrees_north"

        alt_var = ds.createVariable("altitude", "f8", ("file",), fill_value=np.nan)
        alt_var.long_name = "lidar altitude above sea level"
        alt_var.units = "meters"

        zenith_var = ds.createVariable("zenith", "f8", ("file",), fill_value=np.nan)
        zenith_var.long_name = "zenith angle"
        zenith_var.units = "degrees"

        # Laser parameters
        for laser in (1, 2, 3):
            ns = ds.createVariable(
                f"laser{laser}_nshots", "i4", ("file",), fill_value=-1
            )
            ns.long_name = f"laser {laser} number of shots"
            fq = ds.createVariable(f"laser{laser}_freq", "i4", ("file",), fill_value=-1)
            fq.long_name = f"laser {laser} frequency"
            fq.units = "Hz"

        ndatasets_var = ds.createVariable("ndatasets", "i4", ("file",), fill_value=-1)
        ndatasets_var.long_name = "number of datasets (profiles) per file"

        # Write file-level data
        for fi, name in enumerate(file_names):
            lf = pack.Data[name]

            file_name_var[fi] = name
            site_var[fi] = lf.MeasurementSite
            start_time_var[fi] = _dt2ts(lf.MeasurementStartTime)
            stop_time_var[fi] = _dt2ts(lf.MeasurementStopTime)
            lon_var[fi] = lf.Longitude
            lat_var[fi] = lf.Latitude
            alt_var[fi] = lf.AltitudeAboveSeaLevel
            zenith_var[fi] = lf.Zenith
            ndatasets_var[fi] = lf.NDatasets

            ds.variables["laser1_nshots"][fi] = lf.Laser1NShots
            ds.variables["laser1_freq"][fi] = lf.Laser1Freq
            ds.variables["laser2_nshots"][fi] = lf.Laser2NShots
            ds.variables["laser2_freq"][fi] = lf.Laser2Freq
            ds.variables["laser3_nshots"][fi] = lf.Laser3NShots
            ds.variables["laser3_freq"][fi] = lf.Laser3Freq

        # ---- Profile-level variables ----
        prof_file_index_var = ds.createVariable(
            "file_index", "i4", ("profile",), fill_value=-1
        )
        prof_file_index_var.long_name = "index of the parent file"

        wvl_var = ds.createVariable("wavelength", "f8", ("profile",), fill_value=np.nan)
        wvl_var.long_name = "laser wavelength"
        wvl_var.units = "nanometers"

        pol_var = ds.createVariable("polarization", str, ("profile",))
        pol_var.long_name = "polarization channel"

        bw_var = ds.createVariable("bin_width", "f8", ("profile",), fill_value=np.nan)
        bw_var.long_name = "range bin width"
        bw_var.units = "meters"

        nshots_var = ds.createVariable("nshots", "i4", ("profile",), fill_value=-1)
        nshots_var.long_name = "number of laser shots"

        device_id_var = ds.createVariable("device_id", str, ("profile",))
        device_id_var.long_name = "device identifier"

        is_photon_var = ds.createVariable(
            "is_photon", "i4", ("profile",), fill_value=-1
        )
        is_photon_var.long_name = "photon counting channel flag"
        is_photon_var.flag_values = "0, 1"
        is_photon_var.flag_meanings = "analog photon_counting"

        discr_var = ds.createVariable(
            "discr_level", "f8", ("profile",), fill_value=np.nan
        )
        discr_var.long_name = "discriminator level"
        discr_var.units = "millivolts"

        adc_bits_var = ds.createVariable("adc_bits", "i4", ("profile",), fill_value=-1)
        adc_bits_var.long_name = "ADC resolution"

        # Extra profile fields
        active_var = ds.createVariable("active", "i4", ("profile",), fill_value=-1)
        active_var.long_name = "channel active flag"

        laser_type_var = ds.createVariable(
            "laser_type", "i4", ("profile",), fill_value=-1
        )
        laser_type_var.long_name = "laser index"

        high_voltage_var = ds.createVariable(
            "high_voltage", "i4", ("profile",), fill_value=-1
        )
        high_voltage_var.long_name = "PMT high voltage"
        high_voltage_var.units = "volts"

        bin_shift_var = ds.createVariable(
            "bin_shift", "i4", ("profile",), fill_value=-1
        )
        dec_bin_shift_var = ds.createVariable(
            "dec_bin_shift", "i4", ("profile",), fill_value=-1
        )
        n_crate_var = ds.createVariable("n_crate", "i4", ("profile",), fill_value=-1)

        npoints_var = ds.createVariable("npoints", "i4", ("profile",), fill_value=-1)
        npoints_var.long_name = "number of valid data points"

        # ---- Signal data ----
        signal_var = ds.createVariable(
            "signal",
            "f8",
            ("profile", "range"),
            fill_value=np.nan,
            zlib=True,
            complevel=4,
        )
        signal_var.long_name = "lidar signal"
        if profiles[0][1].Photon:
            signal_var.units = "MHz"
        else:
            signal_var.units = "millivolts"
        signal_var.cell_methods = "range: mean"

        # Write profile-level data
        for pi, (fi, p) in enumerate(profiles):
            prof_file_index_var[pi] = fi
            wvl_var[pi] = p.Wavelength
            pol_var[pi] = p.Polarization
            bw_var[pi] = p.BinWidth
            nshots_var[pi] = p.NShots
            device_id_var[pi] = p.DeviceID
            is_photon_var[pi] = 1 if p.Photon else 0
            discr_var[pi] = p.DiscrLevel
            adc_bits_var[pi] = p.AdcBits
            active_var[pi] = 1 if p.Active else 0
            laser_type_var[pi] = p.LaserType
            high_voltage_var[pi] = p.HighVoltage
            bin_shift_var[pi] = p.BinShift
            dec_bin_shift_var[pi] = p.DecBinShift
            n_crate_var[pi] = p.NCrate
            npoints_var[pi] = p.NDataPoints

            signal_var[pi, : p.NDataPoints] = p.Data[: p.NDataPoints]


def from_netcdf(path: str) -> LicelPack:
    """Load a LicelPack from a NetCDF file created by to_netcdf().

    Args:
        path: Path to the .nc file.

    Returns:
        A new LicelPack instance.

    Raises:
        ImportError: If netCDF4 is not installed.
    """
    try:
        from netCDF4 import Dataset
    except ImportError:
        raise ImportError(
            "netCDF4 is required for NetCDF I/O. Install it with: pip install netCDF4"
        )

    ds = Dataset(path, "r")
    try:
        pack = LicelPack()

        n_files = ds.dimensions["file"].size
        n_profiles = ds.dimensions["profile"].size

        signal = ds.variables["signal"][:]

        # Reconstruct files and profiles
        file_map: dict[int, str] = {}

        for fi in range(n_files):
            name = str(ds.variables["file_name"][fi])
            file_map[fi] = name

            lf = LicelFile()
            lf.MeasurementSite = str(ds.variables["site"][fi])
            lf.MeasurementStartTime = _ts2dt(
                float(ds.variables["start_time"][fi])
                if not np.ma.is_masked(ds.variables["start_time"][fi])
                else float("nan")
            )
            lf.MeasurementStopTime = _ts2dt(
                float(ds.variables["stop_time"][fi])
                if not np.ma.is_masked(ds.variables["stop_time"][fi])
                else float("nan")
            )
            lf.Longitude = float(ds.variables["longitude"][fi])
            lf.Latitude = float(ds.variables["latitude"][fi])
            lf.AltitudeAboveSeaLevel = float(ds.variables["altitude"][fi])
            lf.Zenith = float(ds.variables["zenith"][fi])
            lf.Laser1NShots = int(ds.variables["laser1_nshots"][fi])
            lf.Laser1Freq = int(ds.variables["laser1_freq"][fi])
            lf.Laser2NShots = int(ds.variables["laser2_nshots"][fi])
            lf.Laser2Freq = int(ds.variables["laser2_freq"][fi])
            lf.Laser3NShots = int(ds.variables["laser3_nshots"][fi])
            lf.Laser3Freq = int(ds.variables["laser3_freq"][fi])
            lf.NDatasets = int(ds.variables["ndatasets"][fi])
            lf.FileLoaded = True

            pack.Data[name] = lf

        for pi in range(n_profiles):
            fi = int(ds.variables["file_index"][pi])
            p = LicelProfile()
            p.Wavelength = float(ds.variables["wavelength"][pi])
            p.Polarization = str(ds.variables["polarization"][pi])
            p.BinWidth = float(ds.variables["bin_width"][pi])
            p.NShots = int(ds.variables["nshots"][pi])
            p.DeviceID = str(ds.variables["device_id"][pi])
            p.Photon = bool(int(ds.variables["is_photon"][pi]))
            p.DiscrLevel = float(ds.variables["discr_level"][pi])
            p.AdcBits = int(ds.variables["adc_bits"][pi])
            p.Active = bool(int(ds.variables["active"][pi]))
            p.LaserType = int(ds.variables["laser_type"][pi])
            p.HighVoltage = int(ds.variables["high_voltage"][pi])
            p.BinShift = int(ds.variables["bin_shift"][pi])
            p.DecBinShift = int(ds.variables["dec_bin_shift"][pi])
            p.NCrate = int(ds.variables["n_crate"][pi])
            p.NDataPoints = int(ds.variables["npoints"][pi])
            # Reserved is not stored in NC — set to default
            p.Reserved = [0, 0, 0]

            npts = p.NDataPoints
            p.Data = signal[pi, :npts].tolist()

            name = file_map[fi]
            pack.Data[name].Profiles.append(p)

        # Reconstruct StartTime/StopTime from file times
        pack.StartTime = None
        pack.StopTime = None
        for lf in pack.Data.values():
            t = lf.MeasurementStartTime
            if t is None:
                continue
            if pack.StartTime is None or t < pack.StartTime:
                pack.StartTime = t
            if pack.StopTime is None or t > pack.StopTime:
                pack.StopTime = t

    finally:
        ds.close()

    return pack


# ---------------------------------------------------------------------------
# Internal helpers (duplicated to avoid circular imports)
# ---------------------------------------------------------------------------


def _dt2ts(dt: Optional[datetime]) -> float:
    """Convert datetime to Unix timestamp. Returns NaN if None."""
    if dt is None:
        return float("nan")
    return dt.timestamp()


def _ts2dt(ts: float) -> Optional[datetime]:
    """Convert Unix timestamp to datetime. Returns None if NaN."""
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return None
    return datetime.fromtimestamp(ts)
