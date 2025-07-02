import datetime as dt
import numpy as np 
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

import pynasonde.vipir.ngi.utils as utils
from pynasonde.vipir.ngi.source import DataSource
from pynasonde.vipir.ngi.plotlib import Ionogram


def generate_fti_profiles(folder, fig_title="", stn="", flim=[3.5, 4.5]):
    import os

    ds = DataSource(source_folder=folder)
    ds.load_data_sets(0, -1)
    df = 0.3
    for f in np.round(np.arange(2, 4, df), 1):
        os.makedirs(f"tmp/fti/{f}", exist_ok=True)
        # ds.extract_Power_RTI(
        #     folder=f"tmp/fti/{f}", rlim=[95, 110], flim=[f, f+df],
        #     cmap="PuOr", mode="O", fname="FTI.png",
        #     prange=[40, 70], noise_scale=1, del_ticks=False,
        #     interval=1
        # )
        pass
    rti = ds.extract_FTI_RTI(
        rlim=[95, 110],
    )
    fname = f"{ds.datasets[0].URSI}_{rti.time.min().strftime('%Y%m%d.%H%M-')}{rti.time.max().strftime('%H%M')}_O-mode.png"
    fig_title = f"""{ds.datasets[0].URSI}/{rti.time.min().strftime('%H%M-')}{rti.time.max().strftime('%H%M')} UT, {rti.time.max().strftime('%d %b %Y')}"""
    i = Ionogram(fig_title=fig_title, nrows=1, ncols=1)
    ax = i._add_axis(False)
    # ax.set_xlim(xlim)
    ax.set_xlabel("Time, UT", fontdict={"size": i.font_size})
    ax.set_ylim([95, 110])
    ax.set_ylabel("Virtual Height, km", fontdict={"size": i.font_size})
    hours = mdates.HourLocator(interval=1)
    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(DateFormatter(r"$%H^{%M}$"))
    X, Y, Z = utils.get_gridded_parameters(
        rti,
        xparam="time",
        yparam="range",
        zparam="frequency",
        rounding=False,
    )
    im = ax.pcolormesh(
        X,
        Y,
        Z.T,
        cmap="Spectral",
        alpha=0.4,
        zorder=4,
        vmin=1,
        vmax=3,
    )
    i._add_colorbar(im, i.fig, ax, label="")
    i.save(os.path.join("tmp", fname))
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
