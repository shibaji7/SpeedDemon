import sys
import datetime as dt
import os
import numpy as np

sys.path.append("py")
from fetch import create_local_folder, get_ngi_files_by_hour, copy2local, clean_based_on_labels
os.makedirs("figures/traces/", exist_ok=True)

from pynasonde.ngi.source import DataSource
from pynasonde.ngi.utils import load_toml
from pynasonde.ngi.scale import AutoScaler, NoiseProfile
from pynasonde.ngi.plotlib import Ionogram

from pynasonde.polan.datasets import ScaledEntries, ScaledEvent
from pynasonde.polan.polan import Polan

## Copy data to local folder
doy = 236 # Day of year
stn, date = "WI937", dt.datetime(2022, 1, 1) + dt.timedelta(days=doy - 1)
local, remote = create_local_folder(date, stn)
remote_files = get_ngi_files_by_hour(date, [1, 4], remote)
copy2local(local, remote_files)

# Load .ngi folder datasource
ds = DataSource(source_folder=local)
ds.load_data_sets(0, 1) # For now load only one file
cfg = load_toml("py/config.toml") # Load Configuration file, default None

# Run AutoScaler
for i, dx in enumerate(ds.datasets):
    scaler = AutoScaler(
        dx,
        noise_profile=NoiseProfile(constant=cfg.ngi.scaler.noise_constant,),
        mode=cfg.ngi.scaler.mode,
        filter=dict(
            frequency=[cfg.ngi.scaler.frequency_min, cfg.ngi.scaler.frequency_max],
            height=[cfg.ngi.scaler.height_min, cfg.ngi.scaler.height_max],
        ),
        apply_filter=cfg.ngi.scaler.apply_filter,
        segmentation_method=cfg.ngi.scaler.segmentation_method,
    )
    scaler.mdeian_filter()
    scaler.image_segmentation()
    scaler.to_binary_traces(
        nbins=cfg.ngi.scaler.otsu.nbins,
        thresh=cfg.ngi.scaler.otsu.thresh,
        eps=cfg.ngi.scaler.dbscan.eps,
        min_samples=cfg.ngi.scaler.dbscan.min_samples,
    )
    # scaler.draw_sanity_check_images(
    #     f"figures/traces/scan_{dx.time.strftime('%Y%m%d%H%M')}.png", font_size=15
    # )
    
    labels = np.array(scaler.indices.labels.unique()) # Select only > 0 layers tags
    labels = clean_based_on_labels(labels, scaler.indices)
    event = scaler.indices[scaler.indices.labels.isin(labels)] # Select only > 0 layers tags
    # Create Scaling entry parameter
    entries = ScaledEntries(
        date=dx.time,
        events=[
            ScaledEvent(
                description="Parabolic Es + Chapman Layers",
                fv=np.array(event.frequency),  # in MHz
                ht=np.array(event.height),  # in km
                qq=np.empty(50) * 0,
            )
        ]
    )
    p = Polan(entries,)
    sd = p.polan(
        None, dx.time,
        model_ionospheres=[
            dict(
                model="Chapman",
                layer="F",
                Np=9.5e11,
                hp=270,
                scale_h=25,
            ),
            dict(
                model="Parabolic",
                layer="Es",
                D=10,
                hp=102,
                Np=1.7e11,
            ),
        ]
    )
    i = Ionogram(dates=[dx.time], nrows=1, ncols=1, font_size=15, figsize=(7,5))
    ax = i.add_ionogram_traces(
        event.frequency, event.height, del_ticks=False, ylim=[80, 400]
    )
    ax.plot(
        np.log10(sd.fh), sd.h, color="k", ls="-", lw=0.8
    )
    i.add_ionogram_traces(
        sd.tf_sweeps, sd.h_virtual, color="b", ms=3,
        ax=ax, ylim=[80, 400]
    )
    i.save(f"figures/traces/Trace_{dx.time.strftime('%Y%m%d%H%M')}.png")
    i.close()
    del scaler