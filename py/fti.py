import datetime as dt
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.signal import detrend
from loguru import logger
from matplotlib.dates import DateFormatter

import pynasonde.vipir.ngi.utils as utils
from pynasonde.vipir.ngi.source import DataSource
from pynasonde.vipir.ngi.plotlib import Ionogram

import sys
sys.path.append("./py/")
import fetch


# ---------------------------------------------------------------------------
# RTI data builder
# ---------------------------------------------------------------------------

def _build_rti(ds, mode="O", range_lim=(90, 115), freq_lim=(2, 4)):
    """Stack all NGI datasets into a single RTI DataFrame."""
    frames = []
    for dx in ds.datasets:
        time = dt.datetime(dx.year, dx.month, dx.day, dx.hour, dx.minute, dx.second)
        logger.info(f"Time: {time}")
        frequency, rng = np.meshgrid(dx.Frequency, dx.Range, indexing="ij")
        noise, _ = np.meshgrid(
            getattr(dx, f"{mode}_mode_noise"), dx.Range, indexing="ij"
        )
        o = pd.DataFrame({
            "frequency":        frequency.ravel() / 1e3,          # MHz
            "range":            rng.ravel(),                       # km
            f"{mode}_mode_power": getattr(dx, f"{mode}_mode_power").ravel(),  # dB
            f"{mode}_mode_noise": noise.ravel(),                   # dB
            "time":             time,
        })
        o = o[
            (o.range >= range_lim[0]) & (o.range <= range_lim[1]) &
            (o.frequency >= freq_lim[0]) & (o.frequency <= freq_lim[1])
        ]
        frames.append(o)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Lomb-Scargle periodogram
# ---------------------------------------------------------------------------

def lomb_scargle_power_series(
    rti: pd.DataFrame,
    height_km: float = 103.0,
    height_tol_km: float = 5.0,
    freq_lim: tuple = (2.0, 4.0),
    mode: str = "O",
    period_lim_min: tuple = (1.0, 60.0),
    n_periods: int = 1000,
    fname: str = "tmp/jgr_ls.png",
    fig_title: str = "",
    xdate_lims: list = None,
):
    """Compute and plot Lomb-Scargle periodogram of O-mode power at a fixed height.

    Extracts the O-mode power time series at *height_km* (±height_tol_km),
    averages across all frequencies in *freq_lim*, then passes the unevenly-
    sampled series to astropy's LombScargle.

    Parameters
    ----------
    rti          : DataFrame from _build_rti (columns: time, range, O_mode_power, …)
    height_km    : Target virtual height [km].
    height_tol_km: Half-window around height_km for averaging [km].
    freq_lim     : (f_min, f_max) frequency band to average over [MHz].
    mode         : Polarisation mode ('O' or 'X').
    period_lim_min: (T_min, T_max) period range for the periodogram [minutes].
    n_periods    : Number of trial periods.
    fname        : Output PNG path.
    fig_title    : Annotation shown in the top-left of the RTI panel.
    xdate_lims   : [t_start, t_end] datetime limits for the RTI strip.
    """
    power_col = f"{mode}_mode_power"

    # ── Extract power time series at target height ────────────────────────────
    sel = rti[
        (rti.range >= height_km - height_tol_km) &
        (rti.range <= height_km + height_tol_km) &
        (rti.frequency >= freq_lim[0]) &
        (rti.frequency <= freq_lim[1]) 
    ].copy()

    if sel.empty:
        logger.warning(
            f"No data at {height_km} km ± {height_tol_km} km "
            f"in f=[{freq_lim[0]},{freq_lim[1]}] MHz — skipping periodogram"
        )
        return

    # Average power across height window and frequency band per time step
    ts = (
        sel.groupby("time")[power_col]
        .mean()
        .reset_index()
        .sort_values("time")
        .dropna(subset=[power_col])
    )
    if len(ts) < 5:
        logger.warning("Too few time samples for Lomb-Scargle — skipping")
        return

    times_dt = ts["time"].to_numpy()
    power    = ts[power_col].to_numpy(dtype=float)

    # Seconds since first sample (LombScargle needs numeric time)
    t0    = times_dt[0]
    t_sec = (times_dt - t0).astype("timedelta64[s]").astype(float)

    # ── Linear detrend before spectral analysis ───────────────────────────────
    power_det = detrend(power, type="linear")   # remove linear trend/drift

    # ── Lomb-Scargle ──────────────────────────────────────────────────────────
    periods_min  = np.linspace(period_lim_min[0], period_lim_min[1], n_periods)
    frequency_ls = 1.0 / (periods_min * 60.0)           # Hz

    ls       = LombScargle(t_sec, power_det, normalization="psd")
    ls_power = ls.power(frequency_ls)

    # Zoomed LS (1–20 min) for fine structure
    zoom_lim     = (period_lim_min[0], min(20.0, period_lim_min[1]))
    periods_zoom = np.linspace(zoom_lim[0], zoom_lim[1], n_periods)
    ls_power_zoom = ls.power(1.0 / (periods_zoom * 60.0))

    # ── FFT on uniform grid (linearly detrended) ──────────────────────────────
    dt_sec   = float(np.median(np.diff(t_sec))) if len(t_sec) > 1 else 60.0
    t_uni    = np.arange(t_sec[0], t_sec[-1] + dt_sec, dt_sec)
    p_uni    = np.interp(t_uni, t_sec, power_det)
    p_det    = p_uni                                     # already detrended
    n_fft    = len(p_det)
    fft_amp  = np.abs(np.fft.rfft(p_det)) * 2.0 / n_fft
    fft_freq = np.fft.rfftfreq(n_fft, d=dt_sec)           # Hz
    nonzero  = fft_freq > 0
    fft_period_min = 1.0 / fft_freq[nonzero] / 60.0       # minutes
    fft_amp_pos    = fft_amp[nonzero]
    # keep only periods within the requested range
    in_range = (fft_period_min >= period_lim_min[0]) & (fft_period_min <= period_lim_min[1])


    # ── Figure layout ─────────────────────────────────────────────────────────
    # Row 0 (full): RTI (A)
    # Row 1 (full): time series (B)
    # Row 2: LS zoomed 1–20 min (C) | FFT zoomed 1–20 min (E)
    fig = plt.figure(figsize=(8, 7))
    gs  = fig.add_gridspec(
        3, 2,
        height_ratios=[1, 1, 2],
        hspace=0.50, wspace=0.35,
    )
    ax_rti      = fig.add_subplot(gs[0, :])   # A — full-width RTI
    ax_ts       = fig.add_subplot(gs[1, :])   # B — full-width time series
    ax_ls_zoom  = fig.add_subplot(gs[2, 0])   # C — LS zoomed
    ax_fft_zoom = fig.add_subplot(gs[2, 1])   # E — FFT zoomed

    # Panel A — RTI at target height ± tol (pcolormesh)
    piv = (
        sel.pivot_table(index="time", columns="range", values=power_col, aggfunc="mean")
    )
    t_num = mdates.date2num(np.array(piv.index.tolist()))
    h_edges = piv.columns.to_numpy(float)
    T, H = np.meshgrid(t_num, h_edges)
    pcm = ax_rti.pcolormesh(T, H, piv.values.T, cmap="Spectral",
                             vmin=40, vmax=70, shading="nearest")
    pos  = ax_rti.get_position()
    cax  = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.015, pos.height])
    fig.colorbar(pcm, cax=cax, label="O-mode Power, dB")
    ax_rti.axhline(height_km, color="k", ls="--", lw=0.8,
                   label=f"{height_km} km")
    ax_rti.set_ylabel("Virtual Height, km")
    ax_rti.set_ylim(height_km - 10, height_km + 10)
    ax_rti.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax_rti.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
    if xdate_lims:
        ax_rti.set_xlim(mdates.date2num(np.array(xdate_lims)))
    ax_rti.set_title(
        f"(A) RTI @ {height_km} km ± {height_tol_km} km, "
        f"f=[{freq_lim[0]}–{freq_lim[1]}] MHz",
        fontsize=9, loc="left",
    )

    # Panel B — power time series at target height
    ax_ts.plot(times_dt, power, color="steelblue", lw=0.8)
    ax_ts.set_ylabel("Mean Power, dB")
    ax_ts.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
    ax_ts.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
    if xdate_lims:
        ax_ts.set_xlim(xdate_lims)
    ax_ts.set_title(
        f"(B) O-mode power time series at {height_km} km",
        fontsize=9, loc="left",
    )

    # Panel C — LS zoomed (1–20 min)
    ax_ls_zoom.plot(periods_zoom, ls_power_zoom, color="k", lw=0.8)
    ax_ls_zoom.set_xlabel("Period, min")
    ax_ls_zoom.set_ylabel("LS Power (detrended)")
    ax_ls_zoom.set_xlim(zoom_lim)
    ax_ls_zoom.set_ylim(0, 300)
    ax_ls_zoom.set_xticks([2, 5, 8, 12, 15])
    ax_ls_zoom.axvline(7.25, ls="--", lw=0.8, color="m", zorder=5)
    ax_ls_zoom.set_title(f"(C) LS [{zoom_lim[0]}–{zoom_lim[1]} min] — {height_km} km",
                         fontsize=9, loc="left")

    # Panel E — FFT zoomed (1–20 min)
    zoom_mask = (fft_period_min >= zoom_lim[0]) & (fft_period_min <= zoom_lim[1])
    if zoom_mask.any():
        ax_fft_zoom.plot(fft_period_min[zoom_mask], fft_amp_pos[zoom_mask],
                         color="k", lw=0.8)
    ax_fft_zoom.set_xlabel("Period, min")
    ax_fft_zoom.set_ylabel("FFT Amplitude (detrended)")
    ax_fft_zoom.set_xlim(zoom_lim)
    ax_fft_zoom.set_title(f"(E) FFT [{zoom_lim[0]}–{zoom_lim[1]} min] — {height_km} km",
                          fontsize=9, loc="left")
    ax_fft_zoom.set_xticks([2, 5, 8, 12, 15])
    ax_fft_zoom.axvline(7.4, ls="--", lw=0.8, color="m", zorder=5)

    os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {fname}")

    # ── Export time series and periodogram results as CSV ─────────────────────
    csv_base = os.path.splitext(fname)[0]

    # Band A time series
    ts_out = ts.rename(columns={power_col: "mean_power_dB"}).copy()
    ts_out.insert(0, "height_km",   height_km)
    ts_out.insert(1, "freq_min_MHz", freq_lim[0])
    ts_out.insert(2, "freq_max_MHz", freq_lim[1])
    ts_csv = f"{csv_base}_timeseries.csv"
    ts_out.to_csv(ts_csv, index=False)
    logger.info(f"Saved: {ts_csv}")

    # LS + FFT periodogram
    ls_df = pd.DataFrame({
        "period_min":   periods_min,
        "ls_power":     ls_power,
    })
    # align FFT onto same period grid via interpolation (NaN outside FFT range)
    fft_interp = np.where(
        in_range.any() and (fft_period_min[in_range].min() <= periods_min) &
                           (periods_min <= fft_period_min[in_range].max()),
        np.interp(periods_min,
                  fft_period_min[in_range][::-1],   # increasing period → decreasing freq
                  fft_amp_pos[in_range][::-1]),
        np.nan,
    )
    ls_df["fft_amplitude_dB"] = fft_interp
    ls_csv = f"{csv_base}_periodogram.csv"
    ls_df.to_csv(ls_csv, index=False)
    logger.info(f"Saved: {ls_csv}")


# ---------------------------------------------------------------------------
# RTI + jgr.png (original two-panel figure)
# ---------------------------------------------------------------------------

FREQ_BANDS = [
    ("a", (2.0, 2.3)),
    ("b", (2.3, 2.6)),
    ("c", (2.6, 2.9)),
    ("d", (2.9, 3.2)),
]


def generate_fti_profiles(folder, fig_title="", stn=""):
    mode = "O"

    band_csvs = {
        tag: os.path.join("tmp", f"rti_band_{tag}.csv")
        for tag, _ in FREQ_BANDS
    }

    xdate_lims = [
        dt.datetime(2022, 8, 22, 1),
        dt.datetime(2022, 8, 22, 1, 40),
    ]

    if all(os.path.exists(p) for p in band_csvs.values()):
        logger.info("Loading band CSVs from disk (skipping DataSource load)")
        band_dfs = {
            tag: pd.read_csv(p, parse_dates=["time"])
            for tag, p in band_csvs.items()
        }
    else:
        ds = DataSource(source_folder=folder)
        ds.load_data_sets()

        # Per-frequency-band RTI PNGs
        df_step = 0.3
        for f in np.round(np.arange(2, 3.5, df_step), 1):
            os.makedirs(f"tmp/fti/{f}", exist_ok=True)
            ds.extract_Power_RTI(
                folder=f"tmp/fti/{f}", rlim=[95, 110], flim=[f, f + df_step],
                cmap="Spectral", mode=mode, fname="FTI.png",
                prange=[40, 70], noise_scale=1, del_ticks=False,
                xtick_locator=mdates.MinuteLocator(interval=10),
                date_format=r"$%H^{%M}$",
                xdate_lims=[
                    dt.datetime(2022, 8, 22, 1),
                    dt.datetime(2022, 8, 22, 2),
                ],
            )

        rti = _build_rti(ds, mode=mode)
        os.makedirs("tmp", exist_ok=True)
        band_dfs = {}
        for tag, (flo, fhi) in FREQ_BANDS:
            df = rti[(rti.frequency >= flo) & (rti.frequency <= fhi)].copy()
            df.to_csv(band_csvs[tag], index=False)
            logger.info(f"Saved: {band_csvs[tag]}")
            band_dfs[tag] = df

    rti_a = band_dfs["a"]
    rti_b = band_dfs["d"]   # outer bands for jgr.png (2.0-2.3 and 2.9-3.2)

    # ── Two-panel RTI figure (jgr.png) ───────────────────────────────────────
    i = Ionogram(fig_title="", nrows=2, ncols=1, figsize=(6, 3))

    ax = i.add_interval_plots(
        rti_a,
        mode,
        xlabel="", ylabel="Virtual Height, km",
        ylim=[95, 110], add_cbar=False,
        cbar_label="O-mode Power, dB", cmap="Spectral",
        prange=[40, 70], noise_scale=1,
        date_format=r"$%H^{%M}$", del_ticks=False,
        xtick_locator=mdates.MinuteLocator(interval=10),
        xdate_lims=xdate_lims,
    )
    ax.text(0.01, 1.05, fig_title, ha="left", va="center",
            transform=ax.transAxes)
    ax.text(0.95, 1.05, r"(A) $f_0$=[2.0-2.3] MHz", ha="right",
            va="center", transform=ax.transAxes)
    ax.set_xticklabels([])
    for t, ls, lw in [
        (dt.datetime(2022, 8, 22, 1, 16),     "--", 0.5),
        (dt.datetime(2022, 8, 22, 1, 17, 26), "--", 1.5),
        (dt.datetime(2022, 8, 22, 1, 21,  9), "--", 1.5),
    ]:
        ax.axvline(t, ls=ls, lw=lw, color="k" if lw < 1 else "red", zorder=5)

    ax = i.add_interval_plots(
        rti_b,
        mode,
        noise_scale=1, xlabel="Time, UT", ylabel="Virtual Height, km",
        ylim=[95, 110], add_cbar=True,
        cbar_label="O-mode Power, dB", cmap="Spectral",
        prange=[40, 70], date_format=r"$%H^{%M}$", del_ticks=False,
        xtick_locator=mdates.MinuteLocator(interval=10),
        xdate_lims=xdate_lims,
    )
    ax.text(0.95, 1.05, r"(B) $f_0$=[2.9-3.2] MHz",
            ha="right", va="center", transform=ax.transAxes)
    for t, ls, lw in [
        (dt.datetime(2022, 8, 22, 1, 16),     "--", 0.5),
        (dt.datetime(2022, 8, 22, 1, 17, 26), "--", 1.5),
        (dt.datetime(2022, 8, 22, 1, 21,  9), "--", 1.5),
    ]:
        ax.axvline(t, ls=ls, lw=lw, color="k" if lw < 1 else "red", zorder=5)

    i.save(os.path.join("tmp", "jgr.png"))
    i.save(os.path.join("tmp", "jgr.pdf"))
    i.close()

    # ── Lomb-Scargle periodogram (jgr_ls.png) ────────────────────────────────
    rti_all = pd.concat(list(band_dfs.values()), ignore_index=True)
    lomb_scargle_power_series(
        rti_all,
        height_km=102.5,
        height_tol_km=2.0,
        freq_lim=(2., 2.3),
        mode=mode,
        period_lim_min=(2.0, 15.0),
        n_periods=1000,
        fname="tmp/jgr_ls.png",
        fig_title=fig_title,
        xdate_lims=xdate_lims,
    )

    # ── Panel C standalone (jgr_ls_panel.png) ────────────────────────────────
    plot_ls_panel(
        rti_all,
        height_km=102.5,
        height_tol_km=2.0,
        freq_lim=(2., 2.3),
        mode=mode,
        period_lim_min=(2.0, 15.0),
        n_periods=1000,
        fname="tmp/jgr_ls_panel.png",
    )
    return


# ---------------------------------------------------------------------------
# Panel C standalone figure
# ---------------------------------------------------------------------------

def plot_ls_panel(
    rti: pd.DataFrame,
    height_km: float = 102.5,
    height_tol_km: float = 2.0,
    freq_lim: tuple = (2.0, 2.3),
    mode: str = "O",
    period_lim_min: tuple = (2.0, 15.0),
    n_periods: int = 1000,
    fname: str = "tmp/jgr_ls_panel.png",
):
    """Save Panel C (LS zoomed) as a standalone figure."""
    power_col = f"{mode}_mode_power"

    sel = rti[
        (rti["range"] >= height_km - height_tol_km) &
        (rti["range"] <= height_km + height_tol_km) &
        (rti["frequency"] >= freq_lim[0]) &
        (rti["frequency"] <= freq_lim[1])
    ].copy()
    if sel.empty:
        logger.warning("plot_ls_panel: no data in selection — skipping")
        return

    ts = (
        sel.groupby("time")[power_col]
        .mean().reset_index().sort_values("time").dropna(subset=[power_col])
    )
    if len(ts) < 5:
        logger.warning("plot_ls_panel: too few samples — skipping")
        return

    times_dt  = ts["time"].to_numpy()
    power     = ts[power_col].to_numpy(dtype=float)
    t_sec     = (times_dt - times_dt[0]).astype("timedelta64[s]").astype(float)
    power_det = detrend(power, type="linear")

    zoom_lim     = (period_lim_min[0], min(20.0, period_lim_min[1]))
    periods_zoom = np.linspace(zoom_lim[0], zoom_lim[1], n_periods)
    ls_power_zoom = LombScargle(t_sec, power_det, normalization="psd").power(
        1.0 / (periods_zoom * 60.0)
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(periods_zoom, ls_power_zoom, color="k", lw=0.8)
    ax.set_xlabel("Period, min")
    ax.set_ylabel("LS Power (detrended)")
    ax.set_xlim([2, 10])
    ax.set_ylim(0, 200)
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.axvline(7.25, ls="--", lw=0.8, color="m", zorder=5)
    ax.text(7.5, 150, r"$\tau$=7 min 15 sec", ha="left", va="center", color="m", fontsize=9, rotation=90)
    ax.set_title(
        f" LS [{freq_lim[0]}–{freq_lim[1]} MHz] @ {height_km} $\pm${height_tol_km} km",
        fontsize=9, loc="left",
    )

    os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

for doy in range(234, 235):
    stn  = "WI937"
    date = dt.datetime(2022, 1, 1) + dt.timedelta(days=doy - 1)
    print(date)
    fig_title = f"Speed Demon / {stn} / {date.strftime('%Y-%m-%d')}"

    local, remote = fetch.create_local_folder(date=date)
    fetch.copy2local(
        local=local,
        remote_files=fetch.get_ngi_files_by_hour(date=date, hours=[1, 2], remote=remote),
    )

    generate_fti_profiles(folder=local, fig_title=fig_title, stn=stn)
