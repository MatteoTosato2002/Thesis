import subprocess
import os
import re
import warnings
from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
import numpy as np

# Make sure ffmpeg (installed via Homebrew) can be found even if it's not already on PATH.
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"

# ── Helper: builds a "plot spec" (a plain dictionary) instead of drawing a real chart here.
# The GUI (which uses pyqtgraph) reads this dictionary later and draws the actual plot.
def _plot(title: str, *series: Tuple[Any, Any, str, bool]) -> Dict[str, Any]:
    """
    Package one or more data series into a dictionary the GUI can plot later.
    Each series is (x values, y values, label) or (x, y, label, dashed=True).
    """
    specs = []
    for s in series:
        x_raw, y_raw, label = s[0], s[1], s[2]
        dash = s[3] if len(s) > 3 else False
        x = np.array(x_raw, dtype=float)
        y = np.array(y_raw, dtype=float)
        specs.append({'x': x, 'y': y, 'label': label, 'dash': dash})
    return {'title': title, 'series': specs}
# ─────────────────────────────────────────────────────────────────────────────

def estrai_sottotitoli(video_path: str) -> str:
    # If we were given a subtitle file (.srt) directly instead of a video, just use it as-is:
    # no need to run ffmpeg or look for a matching video file.
    if video_path.lower().endswith(".srt"):
        return video_path
    base, _ = os.path.splitext(video_path)
    srt_path = base + ".srt"
    if os.path.exists(srt_path):
        print(f"Subtitles found: {srt_path}")
        return srt_path
    comando = ["ffmpeg", "-i", video_path, "-map", "0:s:0", srt_path, "-y"]
    try:
        subprocess.run(comando, check=True)
        return srt_path
    except Exception as e:
        raise RuntimeError(f"Estrazione SRT fallita per {video_path}: {e}")

# --- READ ALTITUDE FROM A MAVIC DRONE'S SUBTITLE FILE ---
def estrai_H_mavic(srt_path: str, offset_ore: int = -1) -> pd.DataFrame:
    rows = []
    if not os.path.exists(srt_path): return pd.DataFrame()

    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    timecode_pattern = r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->"
    timestamp_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})"

    for block in blocks:
        text = block.replace("\n", " ")
        tc_match = re.search(timecode_pattern, text)
        ts_match = re.search(timestamp_pattern, text)
        h_match = re.search(r"rel_alt:\s*(-?\d+(?:[.,]\d+)?)", text)
        # Mavic subtitle files also carry GPS latitude/longitude for each frame (the older
        # Inspire format doesn't). These are optional: missing them doesn't stop the extraction.
        lat_match = re.search(r"latitude:\s*(-?\d+(?:[.,]\d+)?)", text)
        lon_match = re.search(r"longitude:\s*(-?\d+(?:[.,]\d+)?)", text)

        if tc_match and ts_match and h_match:
            try:
                hh, mm, ss, ms = (int(g) for g in tc_match.groups())
                video_time_s = hh * 3600 + mm * 60 + ss + ms / 1000.0
                ts_raw = pd.to_datetime(ts_match.group(1))
                ts_corrected = ts_raw + pd.Timedelta(hours=offset_ore)
                h_value = float(h_match.group(1).replace(",", "."))
                lat_value = float(lat_match.group(1).replace(",", ".")) if lat_match else np.nan
                lon_value = float(lon_match.group(1).replace(",", ".")) if lon_match else np.nan
                rows.append({
                    "video_time_s": video_time_s, "full_timestamp": ts_corrected, "bar_altitude": h_value,
                    "latitude": lat_value, "longitude": lon_value,
                })
            except Exception: continue

    df = pd.DataFrame(rows)
    if df.empty: return df

    # sec_video counts seconds using the video's OWN internal clock (the "00:00:SS,mmm -->"
    # timecode printed in the subtitles), not the absolute drone clock. The absolute clock
    # almost never starts exactly at .000 seconds, so using it here would shift every video
    # second by up to half a second - small, but visible on a fast-changing altimeter reading.
    df["sec_video"] = df["video_time_s"].astype(int)

    # Absolute timestamp of the frame we picked (needed later to line it up with the laser log).
    # We round to the nearest second instead of always rounding down: a frame at X.998s really
    # belongs to second X+1, and rounding down would wrongly shift the whole sequence back by
    # one second (this happens often on the very first frame of the video).
    df["timestamp"] = df["full_timestamp"].dt.round("s")

    # GROUPING BY SECOND: when a second has several video frames, "first()" keeps the one
    # closest to X.000s (blocks are already in time order, but we sort again just to be safe).
    # We don't bother filtering by "first half of the second" - that would only matter below
    # 2 frames per second, which never happens with real drone footage.
    df_unique = df.sort_values("video_time_s").groupby("sec_video", as_index=False).first()
    df_unique = df_unique.sort_values("sec_video").reset_index(drop=True)[["sec_video", "timestamp", "bar_altitude", "latitude", "longitude"]]

    # sec_video is already guaranteed to count up smoothly (0, 1, 2, 3...), but the absolute
    # clock printed on each frame (used later to line up with the laser log) can occasionally
    # jump forward or backward a bit because of small GPS/RTC glitches - even though the video
    # second itself is still correct and in order. Below we fix ONLY that absolute clock (never
    # sec_video, never the actual data), forcing it to be exactly +1s after the previous
    # corrected value whenever the raw jump isn't exactly 1 second.
    df_unique, timestamp_repair_message = _repair_video_timestamps(df_unique)
    df_unique.attrs["timestamp_repair_message"] = timestamp_repair_message
    return df_unique

# Fixes the small absolute-clock jumps described above and reports what it changed, if anything.
def _repair_video_timestamps(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    ts = df["timestamp"].reset_index(drop=True)
    if len(ts) == 0:
        return df, None
    corrected = [ts.iloc[0]]
    fixed_events = []
    for i in range(1, len(ts)):
        # We always compare to the previous ALREADY-FIXED value, not the raw one: once a
        # jump has been corrected, a "normal-looking" raw value can still be wrong relative
        # to the corrected sequence.
        expected = corrected[-1] + pd.Timedelta(seconds=1)
        if ts.iloc[i] == expected:
            corrected.append(ts.iloc[i])
        else:
            fixed_events.append((ts.iloc[i], expected))
            corrected.append(expected)

    df = df.copy()
    df["timestamp"] = corrected

    if not fixed_events:
        return df, None
    FMT = "%Y-%m-%d %H:%M:%S"
    detail = ["Clock video: rilevato un sobbalzo nel timestamp assoluto (non nel secondo di video, che resta corretto):"]
    for raw_ts, new_ts in fixed_events:
        detail.append(f"  - {raw_ts.strftime(FMT)} rimappato a {new_ts.strftime(FMT)}")
    print("\n".join(detail))
    msg = f"⚠️ Clock video: {len(fixed_events)} sobbalzo/i nel timestamp assoluto corretti (il secondo di video resta invariato)."
    return df, msg

# --- READ ALTITUDE FROM A STANDARD (INSPIRE-STYLE) SUBTITLE FILE ---
def estrai_H_standard(srt_path: str) -> pd.DataFrame:
    rows = []
    if not os.path.exists(srt_path): return pd.DataFrame()

    with open(srt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    blocks = re.split(r"\n\s*\n", content.strip())
    
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 3:
            times = lines[1].split(" --> ")
            try:
                h, m, s_ms = times[0].strip().split(":")
                s, ms = s_ms.split(",")
                video_time_s = int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0
                text = " ".join(lines[2:]).replace("\n", " ")
                match = re.search(r"H\s+(-?\d+(?:[.,]\d+)?)", text)
                if match:
                    h_value = float(match.group(1).replace(",", "."))
                    rows.append({"video_time_s": video_time_s, "H_srt": h_value})
            except (ValueError, IndexError):
                continue

    df_h = pd.DataFrame(rows)
    if df_h.empty: return df_h

    # sec_video = whole second from the subtitle file's own timecode. If a second has more
    # than one frame (high frame-rate subtitles), we always keep the first one - the one
    # closest to X.000s - same rule as the Mavic extraction above, never an average of frames.
    df_h["abs_second"] = df_h["video_time_s"].astype(int)
    df_h["sec_video"] = (df_h["abs_second"] - df_h["abs_second"].min()).astype(int)
    df_h = df_h.sort_values("video_time_s").groupby("sec_video", as_index=False).first()
    return df_h[["sec_video", "H_srt"]]

# --- READ THE LASER ALTIMETER LOG FILE ---
def read_log(laser_csv: str) -> pd.DataFrame:
    header_row = None
    delimiter = '\t'
    header_line = None
    
    # First pass: scan the file to find which row is the header and what separates the columns
    with open(laser_csv, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            # Strip outer quotes and extra spaces
            clean_line = line.strip()
            if clean_line.startswith('"') and clean_line.endswith('"'):
                clean_line = clean_line[1:-1]
            
            if "laser_altitude_cm" in clean_line or "laser_cm" in clean_line:
                header_row = i
                header_line = clean_line
                # Figure out whether columns are separated by tabs or commas
                if '\t' in clean_line:
                    delimiter = '\t'
                elif ',' in clean_line:
                    delimiter = ','
                break
    
    if header_row is None: 
        raise RuntimeError("Header Log Missing: No line containing 'laser_altitude_cm' or 'laser_cm' found")

    # Second pass: read the file again, this time parsing every row by hand so quoted fields are handled correctly
    with open(laser_csv, 'r', encoding='utf-8', errors='ignore') as f:
        all_lines = f.readlines()
    
    # Pull the column names out of the header row
    col_line = all_lines[header_row].strip()
    if col_line.startswith('"'):
        col_line = col_line[1:-1]
    col_names = [c.strip().replace('#', '').replace('"', '') for c in col_line.split(delimiter)]
    
    # Read every data row that comes after the header
    data_lines = []
    for line in all_lines[header_row + 1:]:
        line = line.strip()
        if not line:
            continue
        # Strip outer quotes here too, if present
        if line.startswith('"'):
            line = line[1:-1]
        parts = [p.strip().replace('"', '') for p in line.split(delimiter)]
        if len(parts) == len(col_names):
            data_lines.append(parts)
    
    if not data_lines:
        raise RuntimeError(f"No valid data rows found after header line {header_row}.")
    
    # Build a table (DataFrame) from all the rows we just parsed
    log_df = pd.DataFrame(data_lines, columns=col_names)
    
    # Clean up the column names (remove stray spaces)
    log_df.columns = [c.strip() for c in log_df.columns]
    
    col_mapping = {
        "gmt_date": "date", "date": "date", "gmt_time": "time", "time": "time",
        "laser_altitude_cm": "laser_cm", "laser_cm": "laser_cm",
        "gyro_x": "gyro_x", "gyro_y": "gyro_y", "gyro_z": "gyro_z",
        "tilt_deg": "tilt_deg",
        "latitude": "latitude", "longitude": "longitude"
    }
    log_df = log_df.rename(columns=col_mapping)
    
    # Make sure the columns we actually need are present after renaming
    if "date" not in log_df.columns:
        raise RuntimeError(f"Missing 'date' column after mapping. Available: {list(log_df.columns)}")
    if "time" not in log_df.columns:
        raise RuntimeError(f"Missing 'time' column after mapping. Available: {list(log_df.columns)}")
    if "laser_cm" not in log_df.columns:
        raise RuntimeError(f"Missing 'laser_cm' column. Available: {list(log_df.columns)}")
    
    # Make sure date and time are plain text with no stray spaces
    log_df["date"] = log_df["date"].astype(str).str.strip()
    log_df["time"] = log_df["time"].astype(str).str.strip()
    
    # Sanity check: make sure these columns aren't just empty/NaN everywhere
    if (log_df["date"] == "nan").all() or (log_df["date"] == "").all():
        raise RuntimeError("All 'date' values are empty or NaN. Check file format.")
    if (log_df["time"] == "nan").all() or (log_df["time"] == "").all():
        raise RuntimeError("All 'time' values are empty or NaN. Check file format.")
    
    # Combine date + time into one real datetime value
    log_df["datetime"] = pd.to_datetime(log_df["date"] + " " + log_df["time"], errors="coerce")
    
    # Make sure at least some rows produced a valid datetime
    valid_datetime_count = log_df["datetime"].notna().sum()
    if valid_datetime_count == 0:
        raise RuntimeError(f"No valid datetime values found. Check date/time format. Expected 'YYYY/MM/DD HH:MM:SS' format.")
    
    # Convert the laser reading (in cm) from text to a real number
    log_df["laser_cm"] = pd.to_numeric(log_df["laser_cm"], errors="coerce")

    # Same for the optional columns: since we parsed everything by hand above, they arrive
    # as plain text. Without converting them to numbers, any later math (like np.cos on the
    # tilt angle) would crash with a TypeError instead of giving a real number.
    for c in ["gyro_x", "gyro_y", "gyro_z", "tilt_deg", "latitude", "longitude"]:
        if c in log_df.columns:
            log_df[c] = pd.to_numeric(log_df[c], errors="coerce")

    # Drop any row missing a valid datetime or laser reading - we can't use it
    log_df = log_df.dropna(subset=["datetime", "laser_cm"])
    
    if log_df.empty:
        raise RuntimeError(f"No valid data rows after filtering. Check that file has data rows with valid date, time, and laser_cm values.")
    
    log_df["laser_altitude"] = log_df["laser_cm"].astype(float) / 100.0
    
    # Round each timestamp down to the whole second, then check the log for gaps/jumps
    log_df["second"] = log_df["datetime"].dt.floor("s")
    log_df, repair_msg, repair_detail = _repair_log_seconds(log_df)
    log_df.attrs["log_repair_message"] = repair_msg
    log_df.attrs["log_repair_detail"] = repair_detail
    return log_df

def _repair_log_seconds(log_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str], Optional[str]]:
    """
    Check that the log's timestamps go up one second at a time, without inventing missing
    rows or removing duplicates - we only flag where the sequence breaks.

    Whenever a raw second doesn't follow the previous corrected one by exactly +1s, we just
    accept the raw value as the new reference point and note it down. Repeated seconds are
    never forcibly renumbered either.

    Heads up: this can leave duplicate seconds (two rows sharing the same "second") or gaps
    with no row at all. Later, when this data is merged with the DAT flight log on "second",
    a duplicate produces two matching DAT rows, and a gap just leaves that second missing
    instead of a placeholder empty value.
    """
    log_df = log_df.sort_index().reset_index(drop=True)
    raw_seconds = log_df["second"]

    corrected = [raw_seconds.iloc[0]]
    break_events = []  # each entry: (second we expected, second we actually found)
    for i in range(1, len(raw_seconds)):
        expected = corrected[-1] + pd.Timedelta(seconds=1)
        if raw_seconds.iloc[i] != expected:
            break_events.append((expected, raw_seconds.iloc[i]))
        corrected.append(raw_seconds.iloc[i])

    log_df["second"] = corrected
    log_df["datetime"] = log_df["second"]

    msg = None
    detail_msg = None
    if break_events:
        FMT = "%Y-%m-%d %H:%M:%S"
        # Build a line-by-line detail message: the packaged app has no visible terminal, so
        # keep this ready to show in a GUI popup instead of only printing it.
        detail = ["Log laser: rilevata rottura di continuità nei timestamp (dato grezzo mantenuto, nessuna correzione):"]
        for expected, actual in break_events:
            detail.append(f"  - atteso {expected.strftime(FMT)}, trovato {actual.strftime(FMT)}")
        detail.append("Corretto solo nel file prodotto, il log originale non è stato modificato.")
        detail_msg = "\n".join(detail)
        print(detail_msg)

        # The GUI only shows this short one-line summary; the full detail goes in a
        # separate popup (see log_repair_detail).
        msg = f"⚠️ Log laser: {len(break_events)} rottura/e di continuità — dato grezzo mantenuto (dettagli nel popup)."
    return log_df, msg, detail_msg

# --- READ THE DRONE'S FLIGHT LOG (DAT/CSV EXPORT) ---
def read_dat(apr_csv: str, offset_ore: int = 0, dat_in_feet: bool = False) -> pd.DataFrame:
    apr = pd.read_csv(apr_csv)
    if "datetime(utc)" not in apr.columns: raise RuntimeError("No datetime(utc) in DAT.")

    apr["datetime(utc)"] = pd.to_datetime(apr["datetime(utc)"], errors="coerce")
    apr = apr.dropna(subset=["datetime(utc)"])
    if offset_ore != 0:
        apr["datetime(utc)"] = apr["datetime(utc)"] + pd.Timedelta(hours=offset_ore)

    if not apr.empty:
        # Some DAT exports contain a few stray rows with a wrong/garbage year (e.g. GPS not
        # locked yet at boot): keep only the rows from whichever year appears most often.
        apr["year"] = apr["datetime(utc)"].dt.year
        dominant_year = apr["year"].mode().iloc[0]
        apr = apr[apr["year"] == dominant_year].copy()
        apr.drop(columns="year", inplace=True)

    apr["second"] = apr["datetime(utc)"].dt.floor("s")
    apr = apr.sort_values(["second", "datetime(utc)"])

    # One second can contain several samples (the timestamp column only has 1-second
    # resolution, but the flight log is actually recorded faster): we always keep the
    # first one, same rule used for the subtitle extraction above.
    rows = []
    for _, grp in apr.groupby("second", sort=True):
        rows.append(grp.iloc[0])

    apr_mean = pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame(columns=apr.columns)
    col_alt = next((c for c in apr_mean.columns if "height_above_takeoff" in c.lower() or "altitude" in c.lower()), None)
    if col_alt is None:
        raise RuntimeError(f"No altitude column found in DAT. Available: {list(apr_mean.columns)}")
    apr_mean = apr_mean.rename(columns={col_alt: "bar_altitude"})
    if dat_in_feet:
        apr_mean["bar_altitude"] = pd.to_numeric(apr_mean["bar_altitude"], errors="coerce") * 0.3048
    return apr_mean

# --- FUNCTIONS THAT COMBINE (MERGE) THE DIFFERENT DATA SOURCES ---

# Try lining up two signals in time and score how well each possible shift matches
def _scan_alignment_scores(a: pd.Series, b: pd.Series, lag_range: int = 10, min_overlap: int = 5) -> Optional[Dict[str, Any]]:
    """
    We don't auto-align anything here - we just test every possible whole-second shift
    ("laser time offset") between -lag_range and +lag_range, sliding series `a` (the laser
    readings) the exact same way the GUI's offset field would, and measure how well it then
    correlates (Pearson) with `b` (the barometer readings, which we never touch). The
    best_lag we return is already the number the user can type straight into that offset field.
    """
    scores = {}
    # Correlating two shifted, possibly barely-overlapping series can trigger a numpy
    # "invalid value encountered" warning (a correlation with almost no data) - expected
    # here, so we let it print instead of hiding it.
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        for k in range(-lag_range, lag_range + 1):
            shifted = a.shift(k)
            valid = shifted.notna() & b.notna()
            if valid.sum() > min_overlap:
                r = shifted[valid].corr(b[valid])
                if not np.isnan(r):
                    scores[k] = r
    if not scores:
        return None
    best_lag = max(scores, key=scores.get)
    # base_score (the score at offset 0) can be missing: if the real offset is large, shifting
    # by 0 might not leave enough overlapping data points (< min_overlap), even though there's
    # plenty of overlap at the best_lag. We used to give up on the whole scan in that case,
    # hiding a perfectly good best_lag just because the "offset 0" reference wasn't computable.
    return {"base_score": scores.get(0), "best_lag": best_lag, "best_score": scores[best_lag]}

def _alignment_score_message(
    a: pd.Series, b: pd.Series, lag_range: int = 10,
    height_min: Optional[float] = None, height_max: Optional[float] = None,
) -> str:
    # Uses the same filtered data as the "current offset" Pearson score elsewhere: without
    # this height filter, laser readings outside the expected range are just noise that
    # would distort both the offset-0 score and the suggested best lag.
    if height_min is not None or height_max is not None:
        a = a.copy()
        if height_min is not None: a[a < height_min] = np.nan
        if height_max is not None: a[a > height_max] = np.nan
    result = _scan_alignment_scores(a, b, lag_range=lag_range)
    if result is None:
        return "Dati insufficienti per calcolare lo score di allineamento."
    base, best_lag, best = result["base_score"], result["best_lag"], result["best_score"]
    if base is None:
        # Offset 0 doesn't have enough overlapping data for a Pearson score (probably a big
        # clock drift between the two devices): there's no baseline to compare against, but
        # the best_lag we found is still valid.
        return (
            f"Offset 0s non calcolabile (dati insufficienti). Offset migliore trovato tra "
            f"-{lag_range}s e +{lag_range}s: {best_lag:+d}s (Pearson {best:.2f}) — provalo con il laser time offset."
        )
    if best_lag == 0 or best <= base + 0.01:
        return f"Score allineamento (offset 0s): Pearson {base:.2f}. Nessun offset migliore trovato tra -{lag_range}s e +{lag_range}s."
    return (
        f"Score allineamento (offset 0s): Pearson {base:.2f}. Offset migliore trovato: "
        f"{best_lag:+d}s (Pearson {best:.2f}) — provalo con il laser time offset."
    )

# Lines up the laser log with the DAT flight log by matching their timestamps down to the
# whole second.
def merge_log_dat(log: pd.DataFrame, dat: pd.DataFrame, height_min: Optional[float] = None, height_max: Optional[float] = None) -> pd.DataFrame:
    log_repair_message = log.attrs.get("log_repair_message")
    log_repair_detail = log.attrs.get("log_repair_detail")
    m = pd.merge(log, dat, on="second", how="outer").sort_values("second")
    m["sec_rel"] = (m["second"] - m["second"].min()).dt.total_seconds().astype(int)

    cols_to_keep = ["second", "sec_rel", "laser_altitude", "bar_altitude", "datetime"]
    for c in ["longitude", "latitude", "gyro_x", "gyro_y", "gyro_z", "tilt_deg"]:
        if c in m.columns: cols_to_keep.append(c)

    m = m[cols_to_keep]

    # No automatic shift here: bar_altitude_shifted always starts as an exact copy of the raw data.
    m["bar_altitude_shifted"] = m["bar_altitude"]
    m.attrs["alignment_score_message"] = _alignment_score_message(
        m["laser_altitude"], m["bar_altitude"], height_min=height_min, height_max=height_max
    )

    m.attrs["log_repair_message"] = log_repair_message
    m.attrs["log_repair_detail"] = log_repair_detail
    return m

# Lines up the laser log with a Mavic subtitle file's altitude/GPS data by matching their
# absolute timestamps (rounded to the nearest second).
def merge_log_srt(log: pd.DataFrame, srt: pd.DataFrame, height_min: Optional[float] = None, height_max: Optional[float] = None) -> pd.DataFrame:
    log_repair_message = log.attrs.get("log_repair_message")
    log_repair_detail = log.attrs.get("log_repair_detail")
    timestamp_repair_message = srt.attrs.get("timestamp_repair_message")
    if not log.empty: log["merge_time"] = log["second"].dt.round("1s")
    if not srt.empty and "timestamp" in srt.columns:
        srt["merge_time"] = srt["timestamp"].dt.round("1s")
    m = pd.merge(log, srt, on="merge_time", how="outer", suffixes=("_log", "_srt")).sort_values("merge_time").reset_index(drop=True)
    # Both the laser log (dedicated GPS) and the Mavic subtitles can carry latitude/longitude.
    # When BOTH have it, pandas renames the columns "_log"/"_srt" to avoid a clash. We trust
    # the log's GPS first (usually more reliable) and only use the subtitle's GPS to fill in
    # gaps where the log is missing a value, ending up with one single "latitude"/"longitude"
    # column.
    for c in ["latitude", "longitude"]:
        c_log, c_srt = f"{c}_log", f"{c}_srt"
        if c_log in m.columns or c_srt in m.columns:
            primary = m[c_log] if c_log in m.columns else pd.Series(np.nan, index=m.index)
            fallback = m[c_srt] if c_srt in m.columns else pd.Series(np.nan, index=m.index)
            m[c] = primary.fillna(fallback)
            m.drop(columns=[x for x in (c_log, c_srt) if x in m.columns], inplace=True)
    m["sec_rel"] = (m["merge_time"] - m["merge_time"].min()).dt.total_seconds().astype(int)
    m["sec_vid_vanilla"] = np.nan
    f_idx = m["bar_altitude"].first_valid_index()
    if f_idx is not None:
        if "sec_video" in m.columns:
            m.loc[f_idx:, "sec_vid_vanilla"] = m.loc[f_idx:, "sec_video"]
        else:
            m.loc[f_idx:, "sec_vid_vanilla"] = range(len(m) - f_idx)
# Check whether the log gave us a laser altitude column to compare against the barometer
    col_log = "laser_altitude" if "laser_altitude" in m.columns else None

    # No automatic shift here: bar_altitude_shifted always starts as an exact copy of the raw data.
    m["bar_altitude_shifted"] = m["bar_altitude"]

    if col_log and "bar_altitude" in m.columns:
        m.attrs["alignment_score_message"] = _alignment_score_message(
            m[col_log], m["bar_altitude"], height_min=height_min, height_max=height_max
        )
    else:
        m.attrs["alignment_score_message"] = "Nessun log laser disponibile per calcolare lo score di allineamento."

    m.attrs["log_repair_message"] = log_repair_message
    m.attrs["log_repair_detail"] = log_repair_detail
    m.attrs["timestamp_repair_message"] = timestamp_repair_message
    return m

def merge_DAT_SRT_standard(srt_file: str, rp: int, rd: int, merged: pd.DataFrame, offset_ore: int = 0, match_col: str = "bar_altitude_shifted") -> pd.DataFrame:
    # First try the Inspire-style format ("H <value>" in the subtitles). This format has no
    # absolute timestamp, so the only way to line up sec_video with the log/DAT data is to
    # find where the two altitude curves best match in SHAPE (using Pearson correlation).
    df_h = estrai_H_standard(srt_file)
    if df_h.empty and match_col != "bar_altitude_shifted":
        # Mavic subtitles with no DAT log: their absolute timestamps have nothing to be
        # merged against here (no DAT clock), so they go through the same SHAPE match as the
        # Inspire format, against match_col (the laser curve). Only the altitude is used for
        # the match, so we reshape the Mavic frame into the Inspire one.
        df_mav = estrai_H_mavic(srt_file, offset_ore)
        if not df_mav.empty:
            df_h = df_mav.rename(columns={"bar_altitude": "H_srt"})[["sec_video", "H_srt", "latitude", "longitude"]]
    if not df_h.empty:
        # rp/rd (rows to trim from start/end) only affect the SEARCH for the best match -
        # they exclude noisy altitude readings at the very start/end of the subtitles from
        # the comparison. df_h itself (the real, untrimmed data) is what actually gets merged,
        # so sec_video=0 still lines up with the true first row of the subtitles.
        df_h_match = df_h
        if rp > 0: df_h_match = df_h_match.iloc[rp:]
        if rd > 0: df_h_match = df_h_match.iloc[:-rd] if rd < len(df_h_match) else df_h_match.iloc[0:0]
        df_h_match = df_h_match.reset_index(drop=True)
        if df_h_match.empty: raise RuntimeError("SRT row trimming (Rows Before/After) leaves nothing to match.")

        matches = find_top_n_subsequences_by_shape(merged, df_h_match, n_top=1, match_col=match_col)
        if not matches: raise RuntimeError("No match found.")
        best_idx, best_corr = matches[0]

        # best_idx marks where the FIRST row of the trimmed df_h_match landed - which was
        # really row `rp` of the original subtitles. We shift back by rp seconds so that
        # sec_video=0 always points at the true first row of the subtitles, not the row
        # right after the trim.
        sec_start = merged["sec_rel"].iloc[best_idx] - rp
        merged["sec_video"] = merged["sec_rel"] - sec_start
        # Explicit suffixes: if 'merged' already has an H_srt column (this happens in the
        # Animal Pipeline, which re-merges a CSV already produced by run_pipeline), the new
        # one always becomes "H_srt_y" - spelled out here instead of relying on pandas'
        # default naming.
        result = merged.merge(df_h, left_on="sec_video", right_on="sec_video", how="left", suffixes=("", "_y"))
        # Mavic subtitles carry GPS too: if the laser log already has latitude/longitude the
        # log's GPS wins and the subtitle's ("_y") only fills the gaps - same rule as in the
        # timestamp merge below.
        for c in ["latitude", "longitude"]:
            y = f"{c}_y"
            if y in result.columns:
                result[c] = result[c].fillna(result[y])
                result.drop(columns=[y], inplace=True)
        # pandas' .merge() doesn't carry over .attrs metadata, so we set the match score on
        # the final result here, not on 'merged' before merging (it would be lost otherwise).
        against = "laser" if match_col == "laser_altitude" else "dati"
        result.attrs["srt_match_message"] = f"Match SRT↔{against} (shape, Pearson): r = {best_corr:.3f}"
        return result

    # Mavic format ("rel_alt: <value>"): unlike Inspire, every frame carries an absolute
    # timestamp, so we can line up sec_video the same way the Mavic pipeline does elsewhere
    # (merge_log_srt/sync_mavik_srt_log): round to the nearest second (not always down), so
    # a sample at X.6s attaches to second X+1 and one at X.4s stays at X.
    # No Pearson correlation needed here, so rp/rd (row trimming) don't apply - same as in
    # sync_mavik_srt_log, where they're only used for a diagnostic score, never the merge itself.
    df_mavic = estrai_H_mavic(srt_file, offset_ore)
    if df_mavic.empty: raise RuntimeError("No H in SRT (Inspire or Mavic format).")
    df_mavic = df_mavic.rename(columns={"bar_altitude": "H_srt"})
    # merged["second"] is already a real datetime when it comes straight from merge_log_dat,
    # but it's plain text when 'merged' was just read back from a CSV file (the Animal
    # Pipeline re-merging a file that was already produced once).
    df_mavic["merge_time"] = df_mavic["timestamp"].dt.round("1s")
    merged["merge_time"] = pd.to_datetime(merged["second"]).dt.round("1s")
    merged = merged.merge(
        df_mavic[["merge_time", "sec_video", "H_srt", "latitude", "longitude"]], on="merge_time", how="left", suffixes=("", "_y")
    )
    # If 'merged' already had latitude/longitude (from the laser log, via merge_log_dat),
    # that GPS wins; the subtitle's GPS (renamed "_y" here to avoid a clash) only fills in
    # where the log has nothing - same rule as merge_log_srt above. If 'merged' had no GPS
    # at all (no log), there's no name clash, and the subtitle's columns are kept as-is.
    for c in ["latitude", "longitude"]:
        y = f"{c}_y"
        if y in merged.columns:
            merged[c] = merged[c].fillna(merged[y]) if c in merged.columns else merged[y]
            merged.drop(columns=[y], inplace=True)
    result = merged.drop(columns="merge_time")
    # No Pearson score here (we matched by timestamp, not by curve shape), so instead we
    # report how many rows actually matched - this makes it visible if the hour offset is
    # wrong and the two clocks never actually overlap.
    h_col = "H_srt_y" if "H_srt_y" in result.columns else "H_srt"
    matched = int(result[h_col].notna().sum()) if h_col in result.columns else 0
    result.attrs["srt_match_message"] = (
        f"Match SRT↔dati (timestamp, formato Mavic): {matched}/{len(df_mavic)} righe agganciate"
    )
    return result

# Slides df_h's altitude curve over every possible starting position inside merged_df's
# barometer curve, and keeps the n_top positions where the two curves correlate best
# (highest Pearson r) - this is how we find where in time the subtitles line up.
def find_top_n_subsequences_by_shape(merged_df: pd.DataFrame, df_h: pd.DataFrame, n_top: int = 3, min_overlap: int = 5, verbose: bool = True, match_col: str = "bar_altitude_shifted") -> List[Tuple[int, float]]:
    if df_h.empty: return []
    L = len(df_h)
    # match_col is the curve the subtitles are compared against: the DAT barometer normally,
    # the laser altitude when there is no DAT log at all.
    bar = merged_df[match_col].values
    N = len(bar)
    if L < min_overlap: return []
    h = df_h["H_srt"].values
    corr_list = []
    for start in range(0, N):
        win = bar[start:start+L]
        if len(win) < L:
            # Near the very end of the data there aren't enough points left for a full
            # window, so pad it by repeating the last value to keep the comparison fair.
            win = np.pad(win, (0, L - len(win)), mode='edge')
        if np.isnan(win).any() or np.std(win) == 0 or np.std(h) == 0:
            # Skip this position: missing data, or one of the curves is completely flat
            # (correlation would be undefined).
            continue
        corr = np.corrcoef(h, win)[0, 1]
        corr_list.append((start, corr))
    corr_list.sort(key=lambda x: x[1], reverse=True)
    return corr_list[:n_top]

# Convert sensor width (mm) and image width (px) into millimetres of real space per pixel;
# returns None if the numbers don't make sense (e.g. dividing by zero).
def pix_dim(w_img: float, w_sens: float) -> Optional[float]:
    try:
        return float(w_sens)/float(w_img)
    except (ValueError, ZeroDivisionError):
        return None

# --- HELPER: SAVE THE FINAL, TIDY CSV OUTPUT ---
def _save_cleaned_csv(df: pd.DataFrame, output_path: str) -> None:
    # If we have a real datetime column, use it as the canonical "second" column in the output.
    if "datetime" in df.columns:
        df["second"] = df["datetime"]
    
    # Se esiste la colonna fixed, usala come laser_altitude (sostituisce per l'output)
    #if "laser_altitude_fixed" in df.columns:
    #    df["laser_altitude"] = df["laser_altitude_fixed"]

    # Unified column layout: these columns are the standard output for every pipeline variant.
    # latitude/longitude and the diagnostic columns (H_srt, sec_vid_vanilla,
    # sec_video_previous_file) are only added at the end when that particular run produces them.
    target_cols = [
        "second", "sec_rel", "laser_altitude", "bar_altitude",
        "gyro_x", "gyro_y", "gyro_z", "tilt_deg",
        "bar_altitude_shifted", "sec_video",
        "bar_altitude_fixed", "laser_altitude_fixed",
        "H_srt_y", "laser_altitude_source",
        "latitude", "longitude", "H_srt", "sec_vid_vanilla", "sec_video_previous_file",
    ]
    final_cols = [c for c in target_cols if c in df.columns]
    df[final_cols].to_csv(output_path, index=False)

# --- HELPER: EXPORT A SEPARATE CSV WITH THE MEASURED-OBJECT DATA ---
def _export_measurements_csv(
    merged: pd.DataFrame, measurements: List[Dict[str, Any]], sec_column: str, output_csv: str,
    w_img: float = 3840, w_sens: float = 17.3, focal_lenght: float = 24.96,
) -> Optional[str]:
    """
    If the user measured a reference object in the video ("pitrinu" measurements), write a
    second CSV next to the main output. For each measurement it records the altitudes read
    from the already-merged data (laser + barometer/SRT) plus the object's real-world length,
    computed with the same ground-sample-distance formula used elsewhere for the altitude
    correction - just run in the opposite direction here: from altitude + pixels to metres.
    """
    if not measurements:
        return None
    alt_cols = ["laser_altitude", "laser_altitude_fixed", "laser_altitude_source", "bar_altitude", "bar_altitude_fixed", "tilt_deg"]
    p_dim = pix_dim(w_img, w_sens)
    rows = []
    for m in measurements:
        sec = int(m.get("sec_vid", 0))
        mask = merged[sec_column] == sec if sec_column in merged.columns else pd.Series(dtype=bool)
        row = merged.loc[mask.idxmax()] if mask.any() else {}
        px = float(m.get("len_pitrinu_px", 0))
        rec = {"s": sec, "lenght_px": px}
        for c in alt_cols:
            rec[c] = row.get(c, np.nan) if len(row) else np.nan

        # Ground sample distance formula: convert the object's length in pixels into a
        # real-world length in metres, using the altitude, focal length and sensor size.
        altitude_m = rec["laser_altitude_fixed"]
        length_m = np.nan
        if p_dim and focal_lenght and px and pd.notna(altitude_m):
            length_m = px * altitude_m * p_dim / focal_lenght
        rec["length_m"] = length_m

        # Correct for camera tilt: a tilted shot makes the object look shorter than it really is.
        tilt_deg = rec["tilt_deg"]
        rec["length_m_tilt"] = length_m * np.cos(np.radians(tilt_deg)) if pd.notna(length_m) and pd.notna(tilt_deg) else np.nan

        rows.append(rec)
    path = f"{os.path.splitext(output_csv)[0]}_measurements.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path

# ------------------------------------------------------------------
# RUN PIPELINE (Tab 1)
# ------------------------------------------------------------------
# The main, full pipeline (Tab 1 in the GUI): reads the video subtitles, the laser log and
# the DAT flight log, lines them all up in time, and writes one combined CSV.
def run_pipeline(
    video_file: str, laser_csv: str, apr_csv: Optional[str], output_csv: str,
    righe_prima: int = 0, righe_dopo: int = 0, measurements: Optional[List[Dict[str, Any]]] = None,
    w_img: float = 3840, w_sens: float = 17.3, true_length_m: float = 1.95, focal_lenght: float = 24.96,
    altitude_pitrinu: float = 0,
    apply_correction: bool = False,
    laser_time_offset_s: float = 0, height_min: Optional[float] = None, height_max: Optional[float] = None,
    srt_time_offset_s: float = 0, dat_offset_ore: int = 0, srt_offset_ore: int = 0, dat_in_feet: bool = False,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], str]:
    if measurements is None: measurements = []

    srt_file = estrai_sottotitoli(video_file)
    log = read_log(laser_csv)

    no_dat = not apr_csv
    if no_dat:
        # No DAT log: there is no barometer curve to line the subtitles up against, so the
        # SRT altitude is matched by SHAPE (Pearson) against the LASER curve instead - same
        # search as the Inspire format normally does against the DAT, just on another column.
        # Once matched, H_srt takes the barometer's place downstream (see below), so the
        # correction, the fill, the plots and the CSV all work unchanged.
        merged = log.copy()
        merged["sec_rel"] = (merged["second"] - merged["second"].min()).dt.total_seconds().astype(int)
        log_repair_message = log.attrs.get("log_repair_message")
        log_repair_detail = log.attrs.get("log_repair_detail")
        merged = merge_DAT_SRT_standard(
            srt_file, righe_prima, righe_dopo, merged, offset_ore=srt_offset_ore, match_col="laser_altitude"
        )
        merged.attrs["log_repair_message"] = log_repair_message
        merged.attrs["log_repair_detail"] = log_repair_detail
    else:
        apr = read_dat(apr_csv, offset_ore=dat_offset_ore, dat_in_feet=dat_in_feet)

        # Merge on the raw timestamps (never shifted): laser_altitude/bar_altitude must NEVER be
        # overwritten, and sec_rel needs to stay a stable reference that doesn't move with manual
        # offsets (otherwise, since the laser log usually starts before the DAT log, the offset
        # would just cancel itself out).
        merged = merge_log_dat(log, apr, height_min=height_min, height_max=height_max)
        log_repair_message = merged.attrs.get("log_repair_message")
        log_repair_detail = merged.attrs.get("log_repair_detail")
        alignment_score_message = merged.attrs.get("alignment_score_message")
        # offset_ore here is the SUBTITLE clock's hour offset (Mavic format), not the DAT log's:
        # it's a different clock (embedded in the video file, not the flight-log CSV) and can have
        # its own timezone bug.
        merged = merge_DAT_SRT_standard(srt_file, righe_prima, righe_dopo, merged, offset_ore=srt_offset_ore)
        merged.attrs["log_repair_message"] = log_repair_message
        merged.attrs["log_repair_detail"] = log_repair_detail
        merged.attrs["alignment_score_message"] = alignment_score_message

    # Manual subtitle offset: shifts the H_srt altitude series and, together with it,
    # sec_video - otherwise sec_video would still point at the original video second while
    # H_srt has already moved, breaking any export/measurement that uses sec_video to find
    # the right frame. This is separate from laser_time_offset_s, which instead shifts the
    # laser/gyro/GPS columns.
    if srt_time_offset_s != 0 and "H_srt" in merged.columns:
        srt_offset_rows = int(round(srt_time_offset_s))
        merged["H_srt"] = merged["H_srt"].shift(srt_offset_rows)
        if "sec_video" in merged.columns:
            merged["sec_video"] = merged["sec_video"].shift(srt_offset_rows)

    # With no DAT the subtitle altitude IS the barometer series from here on: everything
    # downstream (correction offset, gap fill, plots, CSV) reads bar_altitude/
    # bar_altitude_shifted, and H_srt is the only altitude reference we have.
    if no_dat:
        merged["bar_altitude"] = merged["H_srt"]
        merged["bar_altitude_shifted"] = merged["H_srt"]

    # Pearson score between subtitles (with the manual offset already applied above) and
    # DAT/barometer: recalculated every run, so changing srt_time_offset_s immediately shows
    # whether the score gets better or worse (same idea as offset_match_message for laser<->baro).
    # Skipped with no DAT: there the two series are the same column, so the score would
    # always be 1.0 - the meaningful one is laser<->SRT (offset_match_message below).
    if not no_dat and "H_srt" in merged.columns and "bar_altitude_shifted" in merged.columns:
        tmp_srt_check = merged.dropna(subset=["H_srt", "bar_altitude_shifted"])
        srt_raw_corr = tmp_srt_check["H_srt"].corr(tmp_srt_check["bar_altitude_shifted"]) if len(tmp_srt_check) > 2 else 0.0
        merged.attrs["srt_offset_match_message"] = (
            f"Corrispondenza SRT↔DAT con offset attuale ({srt_time_offset_s:+g}s): "
            f"Pearson r = {srt_raw_corr:.3f}"
        )

    merged["laser_altitude_shifted"] = merged["laser_altitude"] if "laser_altitude" in merged.columns else np.nan
    if laser_time_offset_s != 0:
        offset_rows = int(round(laser_time_offset_s))
        merged["laser_altitude_shifted"] = merged["laser_altitude_shifted"].shift(offset_rows)
        # gyro/tilt/GPS all come from the same laser log, so they must shift together with it.
        other_laser_cols = [c for c in ["gyro_x", "gyro_y", "gyro_z", "tilt_deg", "latitude", "longitude"] if c in merged.columns]
        for c in other_laser_cols:
            merged[c] = merged[c].shift(offset_rows)

    # Apply height filters if specified (only to laser data)
    if height_min is not None:
        merged.loc[merged["laser_altitude_shifted"] < height_min, "laser_altitude_shifted"] = np.nan
    if height_max is not None:
        merged.loc[merged["laser_altitude_shifted"] > height_max, "laser_altitude_shifted"] = np.nan

    # Pearson score between laser (with the current manual offset already applied) and
    # barometer: recalculated every run, so changing laser_time_offset_s immediately shows
    # whether the score gets better or worse.
    tmp_check = merged.dropna(subset=["laser_altitude_shifted", "bar_altitude_shifted"])
    raw_corr = tmp_check["laser_altitude_shifted"].corr(tmp_check["bar_altitude_shifted"]) if len(tmp_check) > 2 else 0.0
    merged.attrs["offset_match_message"] = (
        f"Corrispondenza laser↔baro con offset attuale ({laser_time_offset_s:+g}s): "
        f"Pearson r = {raw_corr:.3f}"
    )

    figs = []
    series1 = [
        (merged["sec_rel"], merged["laser_altitude_shifted"], "Laser"),
        (merged["sec_rel"], merged["bar_altitude_shifted"], "Baro"),
    ]
    if "H_srt" in merged.columns:
        series1.append((merged["sec_rel"], merged["H_srt"], "SRT (Synced)", True))
    figs.append(_plot("Alignments (Laser vs Baro vs SRT)", *series1))

    # Start the "fixed" column as an exact copy of the original data
    merged["laser_altitude_fixed"] = merged["laser_altitude_shifted"].copy()
    merged["bar_altitude_fixed"] = merged["bar_altitude_shifted"].copy() # Base start
    merged["laser_altitude_source"] = np.where(merged["laser_altitude_fixed"].notna(), "laser", None)

    correction_performed = False
    mean_DHB = 0.0

    # 1. Work out the average difference between the laser (where valid) and the shifted barometer
    mask_val = merged["laser_altitude_shifted"].notna() & merged["bar_altitude_shifted"].notna()
    diff_laser_baro = 0.0
    if mask_val.any():
        diff_laser_baro = (merged.loc[mask_val, "laser_altitude_shifted"] - merged.loc[mask_val, "bar_altitude_shifted"]).mean()

    # 2. Offset logic: if the user measured a reference object, that offset is always
    # calculated and applied to the barometer, regardless of the "Apply Correction & Fill"
    # checkbox. The fallback (average laser-barometer difference, used when there's no
    # measurement) only kicks in when that checkbox is ticked - it's just a rough
    # approximation used to help fill in gaps, not something the user explicitly chose.
    if measurements:
        p_dim = pix_dim(w_img, w_sens)
        true_mm = true_length_m * 1000.0
        DHB_l = []

        for m in measurements:
            px = float(m.get("len_pitrinu_px", 0))
            sec = int(m.get("sec_vid", 0))
            if px <= 0 or not p_dim: continue

            mask = merged["sec_video"] == sec
            if mask.sum() == 0: continue
            row = merged.loc[mask.idxmax()]

            bar_v = row.get("bar_altitude_shifted", np.nan)
            if np.isnan(bar_v): continue

            gsd_real = true_mm / px
            real_mm = ((gsd_real * focal_lenght)/p_dim) + (altitude_pitrinu*1000)
            real_m = real_mm / 1000.0

            DHB_l.append(real_m - bar_v)

        if DHB_l:
            mean_DHB = np.mean(DHB_l)
            correction_performed = True
        elif apply_correction:
            # There are measurements listed but none produced a usable number, and the user still wants the fill
            mean_DHB = diff_laser_baro
            correction_performed = True
    elif apply_correction:
        # No measurements: fall back to the average laser-barometer difference, but only if the user asked for the fill
        mean_DHB = diff_laser_baro
        correction_performed = True

    # 3. Apply the calculated offset (from measurements, or the average) to the barometer
    merged["bar_altitude_fixed"] = merged["bar_altitude_shifted"] + mean_DHB

    # 4. FILL IN MISSING LASER READINGS - only if the user ticked "Apply Correction & Fill"
    if apply_correction:
        # Fill ONLY the gaps in the "fixed" column, using the just-corrected barometer reading
        mask_nan = merged["laser_altitude_fixed"].isna() & merged["bar_altitude_fixed"].notna()
        if mask_nan.any():
            # We use bar_altitude_fixed here since it already has the correct offset applied
            merged.loc[mask_nan, "laser_altitude_fixed"] = merged.loc[mask_nan, "bar_altitude_fixed"]
            merged.loc[mask_nan, "laser_altitude_source"] = "baro"

        # Optional final smoothing step (uncomment below if you want to interpolate remaining gaps)
        #merged["laser_altitude_fixed"] = merged["laser_altitude_fixed"].interpolate(limit_direction="both")

    title_corr = f"Corrected Data (Baro correction = {mean_DHB:+.3f} m)" if correction_performed else "Corrected Data"
    series_corr = [
        (merged["sec_rel"], merged["laser_altitude_fixed"], "Laser Fixed (Filled)"),
        (merged["sec_rel"], merged["bar_altitude_fixed"], "Baro Fixed"),
    ]
    if "H_srt" in merged.columns:
        series_corr.append((merged["sec_rel"], merged["H_srt"], "SRT (Synced)", True))
    figs.append(_plot(title_corr, *series_corr))

    _save_cleaned_csv(merged, output_csv)
    _export_measurements_csv(merged, measurements, "sec_video", output_csv, w_img=w_img, w_sens=w_sens, focal_lenght=focal_lenght)
    return merged, figs, output_csv

# Lines up a SECOND video (e.g. filming the animal instead of the drone's own view) against
# an already-merged CSV produced earlier by run_pipeline.
def run_pipeline_video_sync(video_file_a: str, merged_file: str, output_csv_a: str, righe_prima_animal: int, righe_dopo_animal: int, offset_ore: int = 0, srt_time_offset_s: float = 0) -> Tuple[pd.DataFrame, List[Dict[str, Any]], str]:
    srt = estrai_sottotitoli(video_file_a)
    merged = pd.read_csv(merged_file)
    merged = merge_DAT_SRT_standard(srt, righe_prima_animal, righe_dopo_animal, merged, offset_ore=offset_ore)
    if "H_srt_y" not in merged.columns:
        raise RuntimeError(
            "H_srt_y missing after merge: 'Merged File' must already contain an H_srt column "
            "(output of the Complete Pipeline), otherwise there is no second SRT signal to compare."
        )
    # Manual subtitle offset for the "animal" video: shifts H_srt_y and, together with it,
    # THIS video's sec_video column (never the first video's, already present in the Merged
    # File). The Inspire format overwrites "sec_video" inside merge_DAT_SRT_standard, while
    # the Mavic format produces "sec_video_y" instead (since the first video's "sec_video"
    # already existed before this merge) - we just pick whichever one actually exists.
    if srt_time_offset_s != 0:
        offset_rows = int(round(srt_time_offset_s))
        merged["H_srt_y"] = merged["H_srt_y"].shift(offset_rows)
        sec_video_col = "sec_video_y" if "sec_video_y" in merged.columns else "sec_video"
        if sec_video_col in merged.columns:
            merged[sec_video_col] = merged[sec_video_col].shift(offset_rows)

    # Pearson score between the animal video's subtitles (with the current manual offset
    # already applied) and DAT/barometer: recalculated every run, same idea as the main
    # subtitle score in run_pipeline (srt_offset_match_message).
    if "bar_altitude_shifted" in merged.columns:
        tmp_srt_check = merged.dropna(subset=["H_srt_y", "bar_altitude_shifted"])
        srt_raw_corr = tmp_srt_check["H_srt_y"].corr(tmp_srt_check["bar_altitude_shifted"]) if len(tmp_srt_check) > 2 else 0.0
        merged.attrs["srt_offset_match_message"] = (
            f"Corrispondenza SRT (animal)↔DAT con offset attuale ({srt_time_offset_s:+g}s): "
            f"Pearson r = {srt_raw_corr:.3f}"
        )

    figs = [_plot("Animal Sync",
        (merged["sec_rel"], merged.get("laser_altitude", merged["laser_altitude"]), "Laser"),
        (merged["sec_rel"], merged.get("bar_altitude",   merged["bar_altitude"]),   "Bar"),
        (merged["sec_rel"], merged["H_srt_y"], "H SRT", True),
    )]
    if "bar_altitude_fixed" in merged.columns:
        laser_fixed = merged["laser_altitude_fixed"] if "laser_altitude_fixed" in merged.columns else merged.get("laser_altitude", merged["laser_altitude"])
        figs.append(_plot("Animal Sync (Baro Fixed)",
            (merged["sec_rel"], laser_fixed, "Laser Fixed"),
            (merged["sec_rel"], merged["bar_altitude_fixed"], "Bar Fixed"),
            (merged["sec_rel"], merged["H_srt_y"], "H SRT", True),
        ))
    _save_cleaned_csv(merged, output_csv_a)
    return merged, figs, output_csv_a

# "Phase 1" of the incomplete pipeline (no subtitle/video available yet): merge just the
# laser log and the DAT flight log.
def run_pipeline_no_srt(
    laser_csv: str, apr_csv: str, output_csv: str, laser_time_offset_s: float = 0, offset_ore: int = 0, dat_in_feet: bool = False,
    height_min: Optional[float] = None, height_max: Optional[float] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], str]:
    log = read_log(laser_csv)
    apr = read_dat(apr_csv, offset_ore=offset_ore, dat_in_feet=dat_in_feet)

    # Merge on the raw timestamps (never shifted): laser_altitude/bar_altitude must never be
    # overwritten, and sec_rel stays a stable reference point independent of any manual offset.
    merged = merge_log_dat(log, apr, height_min=height_min, height_max=height_max)

    # Nothing is shifted here: Phase 2 re-reads this CSV and applies its own laser offset to
    # laser/gyro/tilt/GPS. Shifting them here too would move gyro/tilt/GPS twice (laser_altitude
    # is always re-shifted from raw, the others are not). laser_time_offset_s is only a preview
    # on the plot below.
    merged["laser_altitude_shifted"] = merged["laser_altitude"] if "laser_altitude" in merged.columns else np.nan

    # Height filter (laser data only, same rule as run_pipeline).
    if height_min is not None:
        merged.loc[merged["laser_altitude_shifted"] < height_min, "laser_altitude_shifted"] = np.nan
    if height_max is not None:
        merged.loc[merged["laser_altitude_shifted"] > height_max, "laser_altitude_shifted"] = np.nan

    # Same as the other pipeline variants: laser_altitude_fixed must reflect the shift/filter
    # applied above, otherwise _save_cleaned_csv would leave it out of the CSV entirely
    # (the column simply wouldn't exist).
    merged["laser_altitude_fixed"] = merged["laser_altitude_shifted"].copy()

    f = _plot("Base Incomplete",
        (merged["sec_rel"], merged["laser_altitude_shifted"].shift(int(round(laser_time_offset_s))), "Laser"),
        (merged["sec_rel"], merged["bar_altitude_shifted"], "Baro"),
    )
    _save_cleaned_csv(merged, output_csv)
    return merged, [f], output_csv

# "Phase 2" of the incomplete pipeline: apply the laser/barometer offset correction and
# fill to a file already produced by run_pipeline_no_srt.
def run_pipeline_no_srt_correction(
    merged_path: str, target_sec: int, target_val: int, output_csv: str,
    measurements: Optional[List[Dict[str, Any]]] = None, w_img: float = 3840, w_sens: float = 17.3, true_length_m: float = 1.95,
    focal_lenght: float = 24.96, altitude_pitrinu: float = 0,
    apply_correction: bool = False, laser_time_offset_s: float = 0,
    height_min: Optional[float] = None, height_max: Optional[float] = None
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], str]:
    merged = pd.read_csv(merged_path)

    # No automatic shift here: just an informational score (-10s..+10s around offset 0) on
    # the raw data, to flag whether a different offset would line things up better.
    if "laser_altitude" in merged.columns and "bar_altitude" in merged.columns:
        merged.attrs["alignment_score_message"] = _alignment_score_message(
            merged["laser_altitude"], merged["bar_altitude"], height_min=height_min, height_max=height_max
        )

    # At this point 'merged' already has one row per second (no more raw timestamps), so the
    # laser offset can only mean shifting the laser-log columns by N rows/seconds relative to
    # the rest of the already-merged table. laser_altitude and bar_altitude are the raw data
    # and must NEVER be overwritten - the shift and height filter work on
    # laser_altitude_shifted instead, which is what everything from here on uses.
    merged["laser_altitude_shifted"] = merged["laser_altitude"] if "laser_altitude" in merged.columns else np.nan
    if laser_time_offset_s != 0:
        offset_rows = int(round(laser_time_offset_s))
        merged["laser_altitude_shifted"] = merged["laser_altitude_shifted"].shift(offset_rows)
        other_laser_cols = [c for c in ["gyro_x", "gyro_y", "gyro_z", "tilt_deg", "latitude", "longitude"] if c in merged.columns]
        for c in other_laser_cols:
            merged[c] = merged[c].shift(offset_rows)

    # Apply height filters if specified (only to laser data, on the shifted copy)
    if height_min is not None:
        merged.loc[merged["laser_altitude_shifted"] < height_min, "laser_altitude_shifted"] = np.nan
    if height_max is not None:
        merged.loc[merged["laser_altitude_shifted"] > height_max, "laser_altitude_shifted"] = np.nan

    # Pearson score between laser (with the current manual offset already applied) and
    # barometer: recalculated every run, so changing laser_time_offset_s immediately shows
    # whether the score gets better or worse.
    bar_col = "bar_altitude_shifted" if "bar_altitude_shifted" in merged.columns else "bar_altitude"
    if bar_col in merged.columns:
        tmp_check = merged.dropna(subset=["laser_altitude_shifted", bar_col])
        raw_corr = tmp_check["laser_altitude_shifted"].corr(tmp_check[bar_col]) if len(tmp_check) > 2 else 0.0
        merged.attrs["offset_match_message"] = (
            f"Corrispondenza laser↔baro con offset attuale ({laser_time_offset_s:+g}s): "
            f"Pearson r = {raw_corr:.3f}"
        )

    matches = np.where(merged['sec_rel'] == target_sec)[0]
    figures = []
    if len(matches) > 0:
        idx = matches[0]
        merged["sec_video"] = target_val + (np.arange(len(merged)) - idx)

    # Start the "fixed" column as a copy of the shifted/filtered version, not the raw data
    merged["laser_altitude_fixed"] = merged["laser_altitude_shifted"].copy()
    merged["laser_altitude_source"] = np.where(merged["laser_altitude_fixed"].notna(), "laser", None)

    if "bar_altitude_shifted" in merged.columns:
        merged["bar_altitude_fixed"] = merged["bar_altitude_shifted"].copy()
    else:
        merged["bar_altitude_fixed"] = np.nan

    correction_performed = False
    mean_DHB = 0.0

    # 1. Work out the average difference between laser and barometer
    diff_laser_baro = 0.0
    if "bar_altitude_shifted" in merged.columns:
        mask_v = merged["laser_altitude_shifted"].notna() & merged["bar_altitude_shifted"].notna()
        if mask_v.any():
            diff_laser_baro = (merged.loc[mask_v, "laser_altitude_shifted"] - merged.loc[mask_v, "bar_altitude_shifted"]).mean()
    
    # 2. Offset logic: if the user measured a reference object, that offset is always
    # calculated, regardless of the "Apply Correction & Fill" checkbox. The fallback
    # (average laser-barometer difference, used when there's no measurement) only applies
    # when that checkbox is ticked - it's just an approximation used to help the fill.
    if measurements:
        p_dim = pix_dim(w_img, w_sens)
        true_mm = true_length_m * 1000.0
        DHB_l = []
        for m in measurements:
            px = float(m.get("len_pitrinu_px", 0))
            sec = int(m.get("sec_vid", 0))
            if px <= 0: continue
            mask = merged["sec_video"] == sec
            if mask.sum() == 0: continue
            row = merged.loc[mask.idxmax()]
            bar_v = row.get("bar_altitude_shifted", np.nan)
            if np.isnan(bar_v): continue
            gsd_real = true_mm / px
            real_mm = ((gsd_real * focal_lenght)/p_dim) + (altitude_pitrinu*1000)
            real_m = real_mm / 1000.0
            DHB_l.append(real_m - bar_v)
        if DHB_l:
            mean_DHB = np.mean(DHB_l)
            correction_performed = True
        elif apply_correction:
            mean_DHB = diff_laser_baro
            correction_performed = True
    elif apply_correction:
        # No measurements at all: fall back to the average laser-barometer difference
        mean_DHB = diff_laser_baro
        correction_performed = True

    # 3. Apply the correction
    if "bar_altitude_shifted" in merged.columns:
        merged["bar_altitude_fixed"] = merged["bar_altitude_shifted"] + mean_DHB

    # 4. FILL IN MISSING LASER READINGS - only if the user ticked "Apply Correction & Fill"
    if apply_correction and "bar_altitude_fixed" in merged.columns:
        mask_nan = merged["laser_altitude_fixed"].isna() & merged["bar_altitude_fixed"].notna()
        if mask_nan.any():
            # Fill the gaps using the barometer data, already shifted to match the laser's timing
            merged.loc[mask_nan, "laser_altitude_fixed"] = merged.loc[mask_nan, "bar_altitude_fixed"]
            merged.loc[mask_nan, "laser_altitude_source"] = "baro"

    title_corr = f"Corrected No SRT (Baro correction = {mean_DHB:+.3f} m)" if correction_performed else "Corrected No SRT"
    series_corr = [(merged["sec_rel"], merged.get("laser_altitude_fixed", np.full(len(merged), np.nan)), "Laser Fixed")]
    if "bar_altitude_fixed" in merged.columns:
        series_corr.append((merged["sec_rel"], merged["bar_altitude_fixed"], "Baro Fixed"))
    f = _plot(title_corr, *series_corr)
    _save_cleaned_csv(merged, output_csv)
    _export_measurements_csv(merged, measurements, "sec_video", output_csv, w_img=w_img, w_sens=w_sens, focal_lenght=focal_lenght)
    return merged, [f], output_csv

def run_pipeline_no_srt_final_sync(merged_path: str, target_sec: int, target_val: int, output_csv: str) -> str:
    merged = pd.read_csv(merged_path)
    # The sec_video column from the loaded file (Phase 2) needs to be kept under a different
    # name before we write this final phase's new sec_video.
    if "sec_video" in merged.columns:
        merged = merged.rename(columns={"sec_video": "sec_video_previous_file"})
    matches = np.where(merged['sec_rel'] == target_sec)[0]
    if len(matches) > 0:
        idx = matches[0]
        merged['sec_video'] = target_val + (np.arange(len(merged)) - idx)
    _save_cleaned_csv(merged, output_csv)
    return output_csv

# --- SRT <-> DAT (sync_srt_with_dat): no longer used by the main pipeline (SRT/DAT
# merging now goes through merge_DAT_SRT_standard, which handles both Inspire and Mavic).
# Kept commented out instead of deleted: new_GUI.py still references it (run_sync_srt,
# the "SRT <-> DAT" tab) - if it's ever needed again, uncomment it AND add the Mavic case
# (missing here - only the Inspire format via estrai_H_standard is handled).
#
# def sync_srt_with_dat(video_file: str, apr_csv: str, output_csv: str, righe_prima: int, righe_dopo: int, height_min: Optional[float] = None, height_max: Optional[float] = None, offset_ore: int = 0, dat_in_feet: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
#     srt = estrai_sottotitoli(video_file)
#     apr = read_dat(apr_csv, offset_ore=offset_ore, dat_in_feet=dat_in_feet)
#     h = estrai_H_standard(srt)
#
#     # righe_prima/righe_dopo tagliano righe SOLO per la ricerca del match (il Pearson
#     # dentro find_top_n_subsequences_by_shape). h (dati veri, mai tagliati) resta quello
#     # che viene mergiato: sec_video=0 continua a coincidere con la prima riga vera dell'SRT.
#     h_match = h
#     if righe_prima: h_match = h_match.iloc[righe_prima:]
#     if righe_dopo: h_match = h_match.iloc[:-righe_dopo] if righe_dopo < len(h_match) else h_match.iloc[0:0]
#     h_match = h_match.reset_index(drop=True)
#     if h_match.empty: raise RuntimeError("SRT row trimming (Rows Before/After) leaves nothing to match.")
#
#     # Apply height filters if specified (only to laser data, not to SRT)
#     # In this function, laser_altitude is not available, so no filtering applied
#
#     apr["sec_rel"] = (apr["second"] - apr["second"].min()).dt.total_seconds().astype(int)
#     tmp = apr.rename(columns={"bar_altitude": "bar_altitude_shifted"})
#
#     matches = find_top_n_subsequences_by_shape(tmp, h_match, n_top=1)
#     if not matches: raise RuntimeError("No match")
#     idx, _ = matches[0]
#     # idx e' la posizione della PRIMA riga di h_match, cioe' della riga `righe_prima`
#     # dell'SRT originale: correggiamo indietro cosi' sec_video=0 punta sempre alla riga 0
#     # vera dell'SRT, non alla riga dopo il taglio.
#     start = tmp["sec_rel"].iloc[idx] - righe_prima
#     apr["sec_video"] = apr["sec_rel"] - start
#     merged = apr.merge(h, left_on="sec_video", right_on="sec_video", how="left")
#
#     f = _plot("SRT vs DAT Sync",
#         (merged["sec_rel"], merged["bar_altitude"], "Baro"),
#         (merged["sec_rel"], merged["H_srt"],        "H SRT", True),
#     )
#     _save_cleaned_csv(merged, output_csv)
#     return output_csv, [f]

# ------------------------------------------------------------------
# UTILITY: Calculate SRT offset from pixel measurements
# ------------------------------------------------------------------
def calculate_srt_offset_from_measurements(
    merged: pd.DataFrame, measurements: List[Dict[str, Any]], sec_column: str, 
    true_length_m: float, focal_lenght: float, w_img: float, w_sens: float, altitude_pitrinu: float
) -> float:
    """
    Calculate SRT altitude offset based on pixel measurements.
    
    Args:
        merged: DataFrame with merged data
        measurements: List of dicts with 'len_pitrinu_px' and 'sec_video' keys
        sec_column: Column name to match measurements (e.g. 'sec_vid_vanilla' or 'sec_video')
        true_length_m: Known physical length in meters
        focal_lenght: Focal length of camera
        w_img: Image width in pixels
        w_sens: Sensor width in mm
        altitude_pitrinu: Altitude where measurement was taken
    
    Returns:
        offset_srt (float): Calculated offset in meters, rounded to 3 decimals
    """
    p_dim = pix_dim(w_img, w_sens)
    true_mm = true_length_m * 1000.0
    SRT_l = []
    
    for m in measurements:
        px = float(m.get("len_pitrinu_px", 0))
        sec = int(m.get("sec_vid", 0))
        if px <= 0:
            continue
        
        mask = merged[sec_column] == sec
        if mask.sum() == 0:
            continue
        
        row = merged.loc[mask.idxmax()]
        # bar_altitude_shifted only exists when there's a laser log involved (merge_log_srt);
        # without a laser log (merged is just df_srt) the only column available is bar_altitude.
        srt_v = row.get("bar_altitude_shifted", row.get("bar_altitude", np.nan))
        
        # Calculate ground sample distance and real world coordinates
        gsd_real = true_mm / px
        real_mm = ((gsd_real * focal_lenght) / p_dim) + (altitude_pitrinu * 1000)
        real_m = real_mm / 1000.0
        
        if not np.isnan(srt_v):
            SRT_l.append(real_m - srt_v)
    
    return round(float(np.mean(SRT_l)), 3) if SRT_l else 0.0

# ------------------------------------------------------------------
# MAVIC SYNC
# ------------------------------------------------------------------
# Lines up a Mavic drone's subtitle altitude with an optional laser log, with the option to
# fill missing laser readings using the (corrected) subtitle altitude.
def sync_mavik_srt_log(
    srt_file: str, output_csv: str, laser_csv: Optional[str] = None, measurements: Optional[List[Dict[str, Any]]] = None,
    w_img: float = 3840, w_sens: float = 17.29, true_length_m: float = 1.95, focal_lenght: float = 12.1086484375,
    FOV: float = 0.66, altitude_pitrinu: float = 0, offset_ore: int = -1,
    apply_correction: bool = False, laser_time_offset_s: float = 0, height_min: Optional[float] = None, height_max: Optional[float] = None,
    righe_prima: int = 0, righe_dopo: int = 0,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], str]:
    if measurements is None: measurements = []
    if not os.path.exists(srt_file): raise RuntimeError(f"SRT Missing: {srt_file}")

    df_srt = estrai_H_mavic(srt_file, offset_ore)
    if laser_csv == "":
        laser_csv = None

    if laser_csv is not None:
        df_log = read_log(laser_csv)
        # The merge happens on the raw timestamps (never shifted): laser_altitude and
        # bar_altitude are the raw data and must NEVER be overwritten. The manual offset and
        # height filter only act on laser_altitude_shifted, created after the merge, so sec_rel
        # doesn't move together with the log (otherwise the offset would just cancel itself out).
        merged = merge_log_srt(df_log, df_srt, height_min=height_min, height_max=height_max)

        merged["laser_altitude_shifted"] = merged["laser_altitude"] if "laser_altitude" in merged.columns else np.nan
        if laser_time_offset_s != 0:
            offset_rows = int(round(laser_time_offset_s))
            merged["laser_altitude_shifted"] = merged["laser_altitude_shifted"].shift(offset_rows)
            # gyro/tilt/GPS all come from the same laser log, so they must shift together with it.
            other_laser_cols = [c for c in ["gyro_x", "gyro_y", "gyro_z", "tilt_deg", "latitude", "longitude"] if c in merged.columns]
            for c in other_laser_cols:
                merged[c] = merged[c].shift(offset_rows)
        if height_min is not None:
            merged.loc[merged["laser_altitude_shifted"] < height_min, "laser_altitude_shifted"] = np.nan
        if height_max is not None:
            merged.loc[merged["laser_altitude_shifted"] > height_max, "laser_altitude_shifted"] = np.nan

        figs = []
        figs.append(_plot("Raw Data Comparison",
            (merged["sec_rel"], merged["laser_altitude_shifted"],   "Laser (Raw)"),
            (merged["sec_rel"], merged["bar_altitude_shifted"], "Baro/SRT (Raw)"),
        ))

        merged["bar_altitude_fixed"]   = merged["bar_altitude_shifted"].copy()
        merged["laser_altitude_fixed"] = merged["laser_altitude_shifted"].copy()
        merged["laser_altitude_source"] = np.where(merged["laser_altitude_fixed"].notna(), "laser", None)

        # ── Work out the raw laser-vs-subtitle correlation ─────────────────────
        # righe_prima/righe_dopo here only trim the rows used for THIS Pearson score
        # (cutting out noisy readings at the very start/end of the subtitles) - they never
        # touch 'merged' itself, so they don't affect the merge, the fill, or the output CSV.
        # Correlating laser and subtitle altitude over a short/noisy overlap can trigger a
        # numpy "invalid value encountered in true_divide" warning - expected here, so we let
        # it print instead of hiding it.
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            tmp_check = merged.dropna(subset=["laser_altitude_shifted", "bar_altitude_shifted"]).sort_values("merge_time")
            if righe_prima > 0: tmp_check = tmp_check.iloc[righe_prima:]
            if righe_dopo > 0: tmp_check = tmp_check.iloc[:-righe_dopo] if righe_dopo < len(tmp_check) else tmp_check.iloc[0:0]
            raw_corr = float(np.corrcoef(
                tmp_check["laser_altitude_shifted"], tmp_check["bar_altitude_shifted"]
            )[0, 1]) if len(tmp_check) > 2 else 0.0

        # Flag for the GUI: a negative correlation usually means the hour offset is wrong
        merged.attrs["negative_correlation"] = (raw_corr < -0.3)
        merged.attrs["raw_correlation"]       = raw_corr
        # Pearson score between laser (with the current manual offset already applied) and
        # subtitles: recalculated every run, so changing laser_time_offset_s immediately shows
        # whether the score gets better or worse.
        merged.attrs["offset_match_message"] = (
            f"Corrispondenza laser↔SRT con offset attuale ({laser_time_offset_s:+g}s): "
            f"Pearson r = {raw_corr:.3f}"
        )

        # ── Work out the SRT offset: with a measured reference object, the shift is always
        # calculated, regardless of the "Apply Correction & Fill" checkbox. The fallback
        # (average laser-SRT difference, used with no measurements) only applies when that
        # checkbox is ticked - it's just there to help the fill. ─────
        off_srt = 0.0

        if measurements:
            off_srt = calculate_srt_offset_from_measurements(
                merged, measurements, "sec_vid_vanilla",
                true_length_m, focal_lenght, w_img, w_sens, altitude_pitrinu
            )
        elif apply_correction:
            # No measurements: use the average laser-SRT difference only if the user wants the fill
            # Averaging over possibly-empty/all-NaN rows can raise a numpy "mean of empty
            # slice" or invalid-value warning - expected, so let it print instead of hiding it.
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                mask_both = merged["laser_altitude_shifted"].notna() & merged["bar_altitude_shifted"].notna()
                if mask_both.any():
                    off_srt = round(float(
                        (merged.loc[mask_both, "laser_altitude_shifted"] - merged.loc[mask_both, "bar_altitude_shifted"]).mean()
                    ), 3)

        # Apply the offset to the barometer/subtitle series
        merged["bar_altitude_fixed"] = merged["bar_altitude_shifted"] + off_srt

        # ── Fill missing laser readings using the corrected barometer/subtitle data - only
        # if "Apply Correction & Fill" is checked ────
        if apply_correction:
            mask_nan = merged["laser_altitude_fixed"].isna() & merged["bar_altitude_fixed"].notna()
            if mask_nan.any():
                merged.loc[mask_nan, "laser_altitude_fixed"] = merged.loc[mask_nan, "bar_altitude_fixed"]
                merged.loc[mask_nan, "laser_altitude_source"] = "baro"

        title2 = f"Corrected Data (Baro/SRT offset = {off_srt:+.3f} m)" if (measurements or apply_correction) else "Corrected Data"
        figs.append(_plot(title2,
            (merged["sec_rel"], merged["laser_altitude_fixed"], "Laser Fixed (Filled)"),
            (merged["sec_rel"], merged["bar_altitude_fixed"],   "Baro/SRT Fixed"),
        ))
        _export_measurements_csv(merged, measurements, "sec_vid_vanilla", output_csv, w_img=w_img, w_sens=w_sens, focal_lenght=focal_lenght)

    else:
        merged = df_srt
        figs   = []
        figs.append(_plot("Raw Data SRT",
            (merged["sec_video"], merged["bar_altitude"], "Baro/SRT (Raw)"),
        ))

        merged.attrs["negative_correlation"] = False
        off_srt = 0.0

        if measurements:
            off_srt = calculate_srt_offset_from_measurements(
                merged, measurements, "sec_video",
                true_length_m, focal_lenght, w_img, w_sens, altitude_pitrinu
            )
        merged["bar_altitude_fixed"] = merged["bar_altitude"] + off_srt

        title2 = f"Corrected Data (Baro/SRT offset = {off_srt:+.3f} m)" if measurements else "Corrected Data"
        figs.append(_plot(title2,
            (merged["sec_video"], merged["bar_altitude"],       "Baro/SRT vanilla"),
            (merged["sec_video"], merged["bar_altitude_fixed"], "Baro/SRT Fixed"),
        ))
        _export_measurements_csv(merged, measurements, "sec_video", output_csv, w_img=w_img, w_sens=w_sens, focal_lenght=focal_lenght)

    _save_cleaned_csv(merged, output_csv)
    return merged, figs, output_csv


# Same idea as sync_mavik_srt_log above, but the barometer/subtitle offset is a fixed
# number in metres chosen directly by the user, instead of one calculated from a measurement.
def sync_mavik_srt_log_2(
    srt_file: str, output_csv: str, laser_csv: Optional[str] = None, offset_metri: float = 0, offset_ore: int = -1,
    apply_correction: bool = False, laser_time_offset_s: float = 0, height_min: Optional[float] = None, height_max: Optional[float] = None,
    righe_prima: int = 0, righe_dopo: int = 0,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], str]:
    if not os.path.exists(srt_file): raise RuntimeError(f"SRT Missing: {srt_file}")

    df_srt = estrai_H_mavic(srt_file, offset_ore)


    if offset_metri == "":
        offset_metri=0
    if laser_csv == "":
        laser_csv=None

    if laser_csv != None:
        df_log = read_log(laser_csv)
        # Merge on the raw timestamps (never shifted): laser_altitude/bar_altitude must never
        # be overwritten. The manual offset and height filter work on a separate column created
        # after the merge, so sec_rel stays a stable reference point.
        merged = merge_log_srt(df_log, df_srt, height_min=height_min, height_max=height_max)

        merged["laser_altitude_shifted"] = merged["laser_altitude"] if "laser_altitude" in merged.columns else np.nan
        if laser_time_offset_s != 0:
            offset_rows = int(round(laser_time_offset_s))
            merged["laser_altitude_shifted"] = merged["laser_altitude_shifted"].shift(offset_rows)
            # gyro/tilt/GPS all come from the same laser log, so they must shift together with it.
            other_laser_cols = [c for c in ["gyro_x", "gyro_y", "gyro_z", "tilt_deg", "latitude", "longitude"] if c in merged.columns]
            for c in other_laser_cols:
                merged[c] = merged[c].shift(offset_rows)
        if height_min is not None:
            merged.loc[merged["laser_altitude_shifted"] < height_min, "laser_altitude_shifted"] = np.nan
        if height_max is not None:
            merged.loc[merged["laser_altitude_shifted"] > height_max, "laser_altitude_shifted"] = np.nan

        figs = []
        figs.append(_plot("Raw Data Comparison",
            (merged["sec_rel"], merged["laser_altitude_shifted"],       "Laser (Raw)"),
            (merged["sec_rel"], merged["bar_altitude_shifted"], "Baro/SRT (Raw)"),
        ))

        merged["laser_altitude_fixed"] = merged["laser_altitude_shifted"].copy()
        merged["laser_altitude_source"] = np.where(merged["laser_altitude_fixed"].notna(), "laser", None)
        merged["bar_altitude_fixed"] = merged["bar_altitude_shifted"] + offset_metri

        # Pearson score between laser (with the current manual offset already applied) and
        # subtitles: recalculated every run, so changing laser_time_offset_s immediately shows
        # whether the score gets better or worse. righe_prima/righe_dopo only trim the rows used
        # for THIS score (noisy readings at the start/end of the subtitles) - they never touch
        # merged, so they don't affect the merge, the fill, or the output CSV.
        # Correlating two series with almost no overlapping valid points can trigger a numpy
        # "invalid value encountered" warning (division producing NaN) - expected here, so we
        # let it print instead of hiding it.
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            tmp_check = merged.dropna(subset=["laser_altitude_shifted", "bar_altitude_shifted"]).sort_values("merge_time")
            if righe_prima > 0: tmp_check = tmp_check.iloc[righe_prima:]
            if righe_dopo > 0: tmp_check = tmp_check.iloc[:-righe_dopo] if righe_dopo < len(tmp_check) else tmp_check.iloc[0:0]
            raw_corr = float(np.corrcoef(
                tmp_check["laser_altitude_shifted"], tmp_check["bar_altitude_shifted"]
            )[0, 1]) if len(tmp_check) > 2 else 0.0
        merged.attrs["offset_match_message"] = (
            f"Corrispondenza laser↔SRT con offset attuale ({laser_time_offset_s:+g}s): "
            f"Pearson r = {raw_corr:.3f}"
        )

        if apply_correction:
            mask_nan = merged["laser_altitude_fixed"].isna() & merged["bar_altitude_fixed"].notna()
            if mask_nan.any():
                merged.loc[mask_nan, "laser_altitude_fixed"] = merged.loc[mask_nan, "bar_altitude_fixed"]
                merged.loc[mask_nan, "laser_altitude_source"] = "baro"

        figs.append(_plot(f"Corrected Data: offset in meters = {offset_metri}",
            (merged["sec_rel"], merged["laser_altitude_fixed"], "Laser Fixed (Filled)"),
            (merged["sec_rel"], merged["bar_altitude_fixed"],   "Baro/SRT Fixed"),
        ))

    else:
        merged = df_srt
        figs = []
        figs.append(_plot("Raw Data SRT",
            (merged["sec_video"], merged["bar_altitude"], "Baro/SRT (Raw)"),
        ))

        merged["bar_altitude_fixed"] = merged["bar_altitude"] + offset_metri
        figs.append(_plot(f"Corrected Data: offset in meters = {offset_metri}",
            (merged["sec_video"], merged["bar_altitude"],       "Baro/SRT vanilla"),
            (merged["sec_video"], merged["bar_altitude_fixed"], "Baro/SRT Fixed"),
        ))

    _save_cleaned_csv(merged, output_csv)
    return merged, figs, output_csv
