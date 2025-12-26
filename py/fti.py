import datetime as dt
import numpy as np 
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

import pynasonde.vipir.ngi.utils as utils
from pynasonde.vipir.ngi.source import DataSource
from pynasonde.vipir.ngi.plotlib import Ionogram
from scipy.interpolate import griddata

import sys
sys.path.append("./py/")
import fetch

def generate_fti_profiles(folder, fig_title="", stn="", flim=[3.5, 4.5]):
    import os
    import pandas as pd
    from loguru import logger

    ds = DataSource(source_folder=folder)
    ds.load_data_sets()
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
    fig_title = f"WI973/01-02 UT, {date.strftime('%d %b %Y')}"
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
    ax.axvline(dt.datetime(2022, 8, 22, 1, 16), ls="--", lw=0.5, color="k", zorder=5)
    ax.axvline(dt.datetime(2022, 8, 22, 1, 17, 26), ls="--", lw=1.5, color="red", zorder=5)
    ax.axvline(dt.datetime(2022, 8, 22, 1, 21, 9), ls="--", lw=1.5, color="red", zorder=5)

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
    ax.axvline(dt.datetime(2022, 8, 22, 1, 16), ls="--", lw=0.5, color="k", zorder=5)
    ax.axvline(dt.datetime(2022, 8, 22, 1, 17, 26), ls="--", lw=1.5, color="red", zorder=5)
    ax.axvline(dt.datetime(2022, 8, 22, 1, 21, 9), ls="--", lw=1.5, color="red", zorder=5)

    
    i.save(os.path.join("tmp", f"jgr.png"))
    i.close()
    return


## Analyzing the dataset form Speed Deamon 2022
for doy in range(234, 235, 1):
    stn = "WI937"
    date = dt.datetime(2022, 1, 1) + dt.timedelta(days=doy - 1)
    print(date)
    fig_title = f"Speed Demon / {date.strftime('%Y-%m-%d')}"

    local, remote = fetch.create_local_folder(date=date)
    fetch.copy2local(
        local=local,
        remote_files=fetch.get_ngi_files_by_hour(
            date=date,
            hours=[1, 2],
            remote=remote,
        ),
    )

    generate_fti_profiles(
        folder=local,
        fig_title=fig_title,
        stn=stn,
        flim=[2.5, 3.5],
    )
    # shutil.rmtree(f"/tmp/{doy}/", ignore_errors=True)
