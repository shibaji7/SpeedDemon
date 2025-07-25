import datetime as dt
import numpy as np 
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

import pynasonde.vipir.ngi.utils as utils
from pynasonde.vipir.ngi.source import DataSource
from pynasonde.vipir.ngi.plotlib import Ionogram
from scipy.interpolate import griddata


def generate_fti_profiles(folder, fig_title="", stn="", flim=[3.5, 4.5]):
    import os
    import pandas as pd
    from loguru import logger

    ds = DataSource(source_folder=folder)
    ds.load_data_sets(30, 150)
    df = 0.3
    for f in np.round(np.arange(2, 3.5, df), 1):
        os.makedirs(f"tmp/fti/{f}", exist_ok=True)
        ds.extract_Power_RTI(
            folder=f"tmp/fti/{f}", rlim=[95, 110], flim=[f, f+df],
            cmap="Spectral", mode="O", fname="FTI.png",
            prange=[40, 70], noise_scale=1, del_ticks=False,
            xtick_locator=mdates.MinuteLocator(interval=10),
            date_format=r"$%H^{%M}$",
            xdate_lims=[
                dt.datetime(2022, 8, 22, 1),
                dt.datetime(2022, 8, 22, 2),
            ],
        )
        pass
    # rti = ds.extract_FTI_RTI(
    #     rlim=[95, 110],
    # )
    # fname = f"{ds.datasets[0].URSI}_{rti.time.min().strftime('%Y%m%d.%H%M-')}{rti.time.max().strftime('%H%M')}_O-mode.png"
    # fig_title = f"""{ds.datasets[0].URSI}/01-02 UT, {rti.time.max().strftime('%d %b %Y')}"""
    # i = Ionogram(fig_title=fig_title, nrows=1, ncols=1)
    # ax = i._add_axis(False)
    # # ax.set_xlim(xlim)
    # ax.set_xlabel("Time, UT", fontdict={"size": i.font_size})
    # ax.set_ylim([95, 110])
    # ax.set_ylabel("Virtual Height, km", fontdict={"size": i.font_size})
    # hours = mdates.HourLocator(interval=1)
    # ax.xaxis.set_major_locator(hours)
    # ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
    # X, Y, Z = utils.get_gridded_parameters(
    #     rti,
    #     xparam="time",
    #     yparam="range",
    #     zparam="frequency",
    #     rounding=False,
    # )
    # # Mask values above 3.5 MHz
    # Z[Z > 3.5] = np.nan

    # # Spline fill NaNs in Z (X is datetime, so convert to float for interpolation)

    # # Flatten the grids for interpolation
    # X_flat = mdates.date2num(X.flatten())
    # Y_flat = Y.flatten()
    # Z_flat = Z.flatten()

    # # Find valid (non-nan) points
    # mask = ~np.isnan(Z_flat)
    # points = np.column_stack((X_flat[mask], Y_flat[mask]))
    # values = Z_flat[mask]

    # # Interpolate at nan locations
    # nan_mask = np.isnan(Z_flat)
    # interp_points = np.column_stack((X_flat[nan_mask], Y_flat[nan_mask]))

    # if interp_points.size > 0 and points.size > 0:
    #     Z_flat[nan_mask] = griddata(points, values, interp_points, method='cubic')

    # # Reshape back to original grid
    # Z = Z_flat.reshape(Z.shape)
    # im = ax.contourf(
    #     X,
    #     Y,
    #     Z.T,
    #     cmap="jet_r",
    #     alpha=0.4,
    #     vmin=0, vmax=3.5,
    #     levels=np.arange(0, 3.5, .5),
    # )
    # i._add_colorbar(im, i.fig, ax, label=r"$f_0$, MHz")
    # cs = ax.contour(
    #     X,
    #     Y,
    #     Z.T,
    #     levels=np.arange(0, 3.5, .5),
    #     colors='k',
    #     linewidths=0.2,
    #     zorder=5,
    # )
    # # Optionally label the contour lines
    # ax.clabel(cs, inline=True, fontsize=i.font_size * 0.5)
    # ax.set_xlim(dt.datetime(2022, 8, 22, 1), dt.datetime(2022, 8, 22, 2))
    # hours = mdates.MinuteLocator(interval=10)
    # ax.xaxis.set_major_locator(hours)
    # ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
    # ax.axvline(dt.datetime(2022, 8, 22, 1, 16), ls="--", lw=0.5, color="k")

    # i.save(os.path.join("tmp", fname))
    # i.close()


    mode = "O"
    rti = pd.DataFrame()
    for dx in ds.datasets:
        time = dt.datetime(dx.year, dx.month, dx.day, dx.hour, dx.minute, dx.second)
        logger.info(f"Time: {time}")
        frequency, range = np.meshgrid(dx.Frequency, dx.Range, indexing="ij")
        noise, _ = np.meshgrid(
            getattr(dx, f"{mode}_mode_noise"), dx.Range, indexing="ij"
        )
        o = pd.DataFrame()
        (
            o["frequency"],
            o["range"],
            o[f"{mode}_mode_power"],
            o[f"{mode}_mode_noise"],
        ) = (
            frequency.ravel() / 1e3,  # to MHz
            range.ravel(),  # in km
            getattr(dx, f"{mode}_mode_power").ravel(),  # in dB
            noise.ravel(),  # in dB
        )
        o["time"] = time
        o = o[
            (o.range >= 90)
            & (o.range <= 115)
            & (o.frequency >= 2)
            & (o.frequency <= 4)
        ]
        rti = pd.concat([rti, o])
    fig_title = "WI973/01-02 UT, 22 August 2022"
    i = Ionogram(fig_title="", nrows=2, ncols=1, figsize=(6, 3))
    o = rti[
        (rti.frequency >= 2)
        & (rti.frequency <= 2.3)
    ]
    ax = i.add_interval_plots(
        o,
        mode,
        xlabel="",
        ylabel = "Virtual Height, km",
        ylim = [95, 110],
        add_cbar = False,
        cbar_label = "O-mode Power, dB",
        cmap = "Spectral",
        prange = [40, 70],
        noise_scale=1,
        date_format = r"$%H^{%M}$",
        del_ticks = False,
        xtick_locator = mdates.MinuteLocator(interval=10),
        xdate_lims = [
            dt.datetime(2022, 8, 22, 1),
            dt.datetime(2022, 8, 22, 2),
        ],
    )
    ax.text(
        0.01, 1.05, fig_title, ha="left", va="center", transform=ax.transAxes
    )
    ax.text(
        0.95, 1.05, r"(A) $f_0$=[2.0-2.3] MHz", ha="right", va="center", transform=ax.transAxes
    )
    ax.set_xticklabels([])
    ax.axvline(dt.datetime(2022, 8, 22, 1, 16), ls="--", lw=0.5, color="k")
    ax.axvline(dt.datetime(2022, 8, 22, 1, 17, 26), ls="--", lw=1.5, color="red")
    ax.axvline(dt.datetime(2022, 8, 22, 1, 21, 9), ls="--", lw=1.5, color="red")

    flim = [2.9, 3.2]
    o = rti[
        (rti.frequency >= flim[0])
        & (rti.frequency <= flim[1])
    ]
    ax = i.add_interval_plots(
        o,
        mode,
        noise_scale=1,
        xlabel="Time, UT",
        ylabel = "Virtual Height, km",
        ylim = [95, 110],
        cbar_label = "O-mode Power, dB",
        cmap = "Spectral",
        prange = [40, 70],
        date_format = r"$%H^{%M}$",
        del_ticks = False,
        add_cbar = True,
        xtick_locator = mdates.MinuteLocator(interval=10),
        xdate_lims = [
            dt.datetime(2022, 8, 22, 1),
            dt.datetime(2022, 8, 22, 2),
        ],
    )
    ax.text(
        0.95, 1.05, fr"(B) $f_0$=[{flim[0]}-{flim[1]}] MHz", ha="right", va="center", transform=ax.transAxes
    )
    ax.axvline(dt.datetime(2022, 8, 22, 1, 16), ls="--", lw=0.5, color="k")
    ax.axvline(dt.datetime(2022, 8, 22, 1, 17, 26), ls="--", lw=1.5, color="red")
    ax.axvline(dt.datetime(2022, 8, 22, 1, 21, 9), ls="--", lw=1.5, color="red")

    
    i.save(os.path.join("tmp", f"jgr.png"))
    i.close()
    return


## Analyzing the dataset form Speed Deamon 2022
for doy in range(234, 235, 1):
    stn = "WI937"
    date = dt.datetime(2022, 1, 1) + dt.timedelta(days=doy - 1)
    fig_title = f"Speed Demon / {date.strftime('%Y-%m-%d')}"

    generate_fti_profiles(
        folder=f"/tmp/{doy}/",
        fig_title=fig_title,
        stn=stn,
        flim=[2.5, 3.5],
    )
    # shutil.rmtree(f"/tmp/{doy}/", ignore_errors=True)
