"""Reads the drone's flight log: altitude, camera tilt and flight date."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TelemetryData:
    dataframe: "pd.DataFrame"
    available: bool
    seconds: "np.ndarray"
    altitudes: "np.ndarray"
    tilt_available: bool
    tilt_seconds: "np.ndarray | None"
    tilts: "np.ndarray | None"


def load_telemetry(
    csv_path,
    video_time_column,
    altitude_column,
    use_tilt_filter,
    measure_in_meters,
) -> TelemetryData:
    df_telemetry = pd.read_csv(csv_path) if csv_path else pd.DataFrame()
    required_columns = {video_time_column, altitude_column}
    missing_columns = required_columns.difference(df_telemetry.columns)
    telemetry_available = not missing_columns

    if measure_in_meters and not csv_path:
        raise ValueError(
            "To measure in meters you must select the telemetry CSV."
        )
    if measure_in_meters and missing_columns:
        raise ValueError(
            "The telemetry CSV does not contain the required columns: "
            + ", ".join(sorted(missing_columns))
        )
    if csv_path and not telemetry_available:
        print(
            "  > Altitude telemetry not available: the optional CSV "
            "does not contain the configured time and altitude columns."
        )

    if telemetry_available:
        telemetry_clean = (
            df_telemetry[[video_time_column, altitude_column]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .sort_values(video_time_column)
            .drop_duplicates(video_time_column)
        )
        if telemetry_clean.empty:
            if measure_in_meters:
                raise ValueError(
                    "The CSV does not contain valid time and altitude values."
                )
            telemetry_available = False
            telemetry_seconds = np.array([], dtype=float)
            telemetry_altitudes = np.array([], dtype=float)
        else:
            telemetry_seconds = telemetry_clean[video_time_column].to_numpy(dtype=float)
            telemetry_altitudes = telemetry_clean[altitude_column].to_numpy(dtype=float)

            # Tell the user how many rows were unusable.
            discarded = len(df_telemetry) - len(telemetry_clean)
            if discarded:
                print(f"  > Telemetry: discarded {discarded} non-numeric or incomplete rows.")
            if len(telemetry_seconds) > 1:
                largest_gap = float(np.diff(telemetry_seconds).max())
                if largest_gap > 1.0:
                    print(f"  > Warning: largest gap in telemetry is {largest_gap:.0f} s.")
    else:
        telemetry_seconds = np.array([], dtype=float)
        telemetry_altitudes = np.array([], dtype=float)

    tilt_available = telemetry_available and "tilt_deg" in df_telemetry.columns
    telemetry_tilt_seconds = None
    telemetry_tilts = None
    if tilt_available:
        tilt_clean = (
            df_telemetry[[video_time_column, "tilt_deg"]]
            .apply(pd.to_numeric, errors="coerce")
            .dropna()
            .sort_values(video_time_column)
            .drop_duplicates(video_time_column)
        )
        if tilt_clean.empty:
            tilt_available = False
        else:
            telemetry_tilt_seconds = tilt_clean[video_time_column].to_numpy(
                dtype=float
            )
            telemetry_tilts = tilt_clean["tilt_deg"].to_numpy(dtype=float)

    if measure_in_meters and use_tilt_filter and not tilt_available:
        print(
            "  > Warning: tilt filter requested, but the tilt_deg column "
            "is not present or does not contain numeric values. Filter ignored."
        )

    return TelemetryData(
        dataframe=df_telemetry,
        available=telemetry_available,
        seconds=telemetry_seconds,
        altitudes=telemetry_altitudes,
        tilt_available=tilt_available,
        tilt_seconds=telemetry_tilt_seconds,
        tilts=telemetry_tilts,
    )


def _sample_at_or_before(sample_seconds, sample_values, seconds):
    """Value logged at `seconds`, or at the last second before it.

    The log has one row per second, so the frame's second is rounded down and
    the value logged at that second is used as-is.
    """
    if sample_seconds is None or len(sample_seconds) == 0:
        return np.nan
    target_second = float(np.floor(seconds))
    index = int(np.searchsorted(sample_seconds, target_second, side="right")) - 1
    if index < 0:
        return np.nan
    return float(sample_values[index])


def get_altitude(telemetry: TelemetryData, seconds):
    """Altitude at the frame's second (rounded down). NaN without telemetry."""
    if not telemetry.available:
        return np.nan
    return _sample_at_or_before(telemetry.seconds, telemetry.altitudes, seconds)


def get_nearest_altitude(telemetry: TelemetryData, seconds):
    """Old name for get_altitude, kept so existing callers keep working."""
    return get_altitude(telemetry, seconds)


def get_tilt(telemetry: TelemetryData, seconds):
    """Camera tilt at the frame's second (rounded down), if it was logged."""
    if not telemetry.tilt_available:
        return np.nan
    return _sample_at_or_before(telemetry.tilt_seconds, telemetry.tilts, seconds)


def extract_flight_date(dataframe, flight_date_column=""):
    requested_column = flight_date_column.strip()
    normalized_columns = {
        str(column).strip().lower(): column for column in dataframe.columns
    }
    candidates = []
    if requested_column:
        requested_match = normalized_columns.get(requested_column.lower())
        if requested_match is not None:
            candidates.append(requested_match)
        else:
            print(
                f"  > Warning: date column '{requested_column}' not found; "
                "trying automatic detection."
            )

    preferred_names = (
        "second",
        "flight_date",
        "date",
        "data",
        "gps_date",
        "recording_date",
        "datetime",
        "date_time",
        "gps_datetime",
        "utc_datetime",
        "timestamp",
    )
    for preferred in preferred_names:
        column = normalized_columns.get(preferred)
        if column is not None and column not in candidates:
            candidates.append(column)
    for column in dataframe.columns:
        normalized = str(column).strip().lower()
        if (
            ("date" in normalized or "data" in normalized)
            and column not in candidates
        ):
            candidates.append(column)

    for column in candidates:
        values = dataframe[column].dropna()
        for value in values.head(50):
            parsed = pd.NaT
            if isinstance(value, (int, float, np.integer, np.floating)):
                numeric_value = float(value)
                if 20_000 <= numeric_value <= 80_000:
                    # Excel serial date: day 1 is 1899-12-30.
                    parsed = pd.Timestamp("1899-12-30") + pd.to_timedelta(
                        numeric_value,
                        unit="D",
                    )
                elif 10_000_000 <= numeric_value <= 99_999_999:
                    parsed = pd.to_datetime(
                        str(int(numeric_value)),
                        format="%Y%m%d",
                        errors="coerce",
                    )
                elif 1_000_000_000 <= numeric_value < 10_000_000_000:
                    parsed = pd.to_datetime(
                        numeric_value,
                        unit="s",
                        errors="coerce",
                    )
                elif numeric_value >= 1_000_000_000_000:
                    parsed = pd.to_datetime(
                        numeric_value,
                        unit="ms",
                        errors="coerce",
                    )
                if pd.isna(parsed):
                    # Stops plain elapsed seconds (0, 1, 2...) from being read
                    # as dates in 1970.
                    continue
            if pd.isna(parsed):
                # Keep dayfirst=False (pandas' default). With dayfirst=True,
                # pandas swaps day and month even on unambiguous ISO strings:
                # pd.to_datetime("2025-04-05 11:02:57", dayfirst=True) returns
                # 2025-05-04. That really happened, and exported frames ended
                # up named with the wrong month. False is right for ISO
                # (year first, what every date column we have seen uses) and
                # for MM/DD/YYYY. A true DD/MM/YYYY log would need a second
                # look, but has not shown up in the data so far.
                parsed = pd.to_datetime(value, errors="coerce", dayfirst=False)
            if pd.notna(parsed):
                date_label = parsed.strftime("%Y-%m-%d")
                print(
                    f"  > Flight date: {date_label} "
                    f"(column '{column}')."
                )
                return date_label

    print(
        "  > Warning: flight date not recognized in the CSV; "
        "using 'date-unknown' in filenames."
    )
    return "date-unknown"
