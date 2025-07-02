import datetime as dt
import numpy as np 
# import shutil

from pynasonde.vipir.ngi.source import DataSource


def generate_fti_profiles(folder, fig_file_name, fig_title="", stn="", flim=[3.5, 4.5]):
    import os

    os.makedirs(f"/tmp/fti/{flim[0]}", exist_ok=True)
    ds = DataSource(source_folder=folder)
    ds.load_data_sets(0, -1)
    for f in np.arange(2, 4, 0.1):
        os.makedirs(f"/tmp/fti/{f}", exist_ok=True)
        ds.extract_FTI_RTI(folder=f"/tmp/fti/{f}", rlim=[100, 130], flim=[f, f+0.1])
    return


## Analyzing the dataset form Speed Deamon 2022
for doy in range(234, 235, 1):
    stn = "WI937"
    date = dt.datetime(2022, 1, 1) + dt.timedelta(days=doy - 1)
    fig_file_name = f"../../tmp/FTI.{stn}.2022.doy-{doy}.png"
    fig_title = f"Speed Demon / {date.strftime('%Y-%m-%d')}"

    generate_fti_profiles(
        folder=f"/tmp/{doy}/",
        fig_file_name=fig_file_name,
        fig_title=fig_title,
        stn=stn,
        flim=[2.5, 3.5],
    )
    # shutil.rmtree(f"/tmp/{doy}/", ignore_errors=True)
