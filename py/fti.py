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
    # Row 2: LS zoomed 1–20 min (C) | LS full range (D)
    # Row 3: FFT zoomed 1–20 min (E) | FFT full range (F)
    fig = plt.figure(figsize=(10, 12))
    gs  = fig.add_gridspec(
        4, 2,
        height_ratios=[1, 1, 2, 2],
        hspace=0.50, wspace=0.35,
    )
    ax_rti      = fig.add_subplot(gs[0, :])   # A — full-width RTI
    ax_ts       = fig.add_subplot(gs[1, :])   # B — full-width time series
    ax_ls_zoom  = fig.add_subplot(gs[2, 0])   # C — LS zoomed
    ax_ls_full  = fig.add_subplot(gs[2, 1])   # D — LS full range
    ax_fft_zoom = fig.add_subplot(gs[3, 0])   # E — FFT zoomed
    ax_fft_full = fig.add_subplot(gs[3, 1])   # F — FFT full range

    # Panel A — RTI at target height ± tol (pcolormesh)
    piv = (
        sel.pivot_table(index="time", columns="range", values=power_col, aggfunc="mean")
    )
    t_num = mdates.date2num(np.array(piv.index.tolist()))
    h_edges = piv.columns.to_numpy(float)
    T, H = np.meshgrid(t_num, h_edges)
    pcm = ax_rti.pcolormesh(T, H, piv.values.T, cmap="Spectral",
                             vmin=40, vmax=70, shading="nearest")
    plt.colorbar(pcm, ax=ax_rti, label="O-mode Power, dB", pad=0.02)
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

    # ── FAP levels ────────────────────────────────────────────────────────────
    fap_levels = [0.01, 0.001]
    fap_powers = [ls.false_alarm_level(f) for f in fap_levels]
    fap_colors = ["red", "darkorange"]
    fap_labels = ["FAP 1%", "FAP 0.1%"]

    def _decorate_ls(ax, periods, lsp, xlim, label):
        ax.plot(periods, lsp, color="k", lw=0.8)
        for fp, fc, fl in zip(fap_powers, fap_colors, fap_labels):
            ax.axhline(fp, color=fc, ls="--", lw=0.8, label=fl)
        peak_idx = np.argmax(lsp)
        ax.axvline(periods[peak_idx], color="steelblue", ls=":", lw=1.0,
                   label=f"Peak {periods[peak_idx]:.1f} min")
        ax.set_xlabel("Period, min")
        ax.set_ylabel("LS Power (detrended)")
        ax.set_xlim(xlim)
        ax.set_ylim(bottom=0, top=300)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(label, fontsize=9, loc="left")

    # Panel C — LS zoomed (1–20 min)
    _decorate_ls(ax_ls_zoom, periods_zoom, ls_power_zoom,
                 zoom_lim, f"(C) LS zoomed [{zoom_lim[0]}–{zoom_lim[1]} min] — {height_km} km")

    # Panel D — LS full range
    _decorate_ls(ax_ls_full, periods_min, ls_power,
                 period_lim_min, f"(D) LS full [{period_lim_min[0]}–{period_lim_min[1]} min] — {height_km} km")

    # ── FFT panels ────────────────────────────────────────────────────────────
    def _decorate_fft(ax, mask, xlim, label):
        if mask.any():
            ax.plot(fft_period_min[mask], fft_amp_pos[mask], color="k", lw=0.8)
            peak_idx = np.argmax(fft_amp_pos[mask])
            peak_per = fft_period_min[mask][peak_idx]
            ax.axvline(peak_per, color="steelblue", ls=":", lw=1.0,
                       label=f"Peak {peak_per:.1f} min")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("Period, min")
        ax.set_ylabel("FFT Amplitude (detrended)")
        ax.set_xlim(xlim)
        ax.set_ylim(bottom=0, top=5)
        ax.set_title(label, fontsize=9, loc="left")

    zoom_mask = (fft_period_min >= zoom_lim[0]) & (fft_period_min <= zoom_lim[1])
    full_mask  = in_range

    # Panel E — FFT zoomed
    _decorate_fft(ax_fft_zoom, zoom_mask, zoom_lim,
                  f"(E) FFT zoomed [{zoom_lim[0]}–{zoom_lim[1]} min] — {height_km} km")

    # Panel F — FFT full range
    _decorate_fft(ax_fft_full, full_mask, period_lim_min,
                  f"(F) FFT full [{period_lim_min[0]}–{period_lim_min[1]} min] — {height_km} km")

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

def generate_fti_profiles(folder, fig_title="", stn=""):
    mode = "O"

    csv_a = os.path.join("tmp", "rti_band_a.csv")
    csv_b = os.path.join("tmp", "rti_band_b.csv")
    band_a = (2.0, 2.3)
    band_b = (2.9, 3.2)

    xdate_lims = [
        dt.datetime(2022, 8, 22, 1),
        dt.datetime(2022, 8, 22, 1, 40),
    ]

    if os.path.exists(csv_a) and os.path.exists(csv_b):
        logger.info("Loading band CSVs from disk (skipping DataSource load)")
        rti_a = pd.read_csv(csv_a, parse_dates=["time"])
        rti_b = pd.read_csv(csv_b, parse_dates=["time"])
        rti   = pd.concat([rti_a, rti_b], ignore_index=True)
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

        rti   = _build_rti(ds, mode=mode)
        rti_a = rti[(rti.frequency >= band_a[0]) & (rti.frequency <= band_a[1])].copy()
        rti_b = rti[(rti.frequency >= band_b[0]) & (rti.frequency <= band_b[1])].copy()
        os.makedirs("tmp", exist_ok=True)
        rti_a.to_csv(csv_a, index=False)
        rti_b.to_csv(csv_b, index=False)
        logger.info(f"Saved: {csv_a}")
        logger.info(f"Saved: {csv_b}")

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
    ax.text(0.95, 1.05, fr"(B) $f_0$=[{band_b[0]}-{band_b[1]}] MHz",
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
    lomb_scargle_power_series(
        rti,
        height_km=105.0,
        height_tol_km=2.0,
        freq_lim=(2., 2.3),
        mode=mode,
        period_lim_min=(1.0, 20.0),
        n_periods=1000,
        fname="tmp/jgr_ls.png",
        fig_title=fig_title,
        xdate_lims=xdate_lims,
    )
    return


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
