import shutil
import os
import glob
import numpy as np
from loguru import logger
import datetime as dt

def create_local_folder(
    date:dt.datetime, 
    stn:str = "WI937", 
    instr: str = "ionogram",
    base:str="/tmp/chakras4/ERAU/SpeedDemon/",
    clean:bool=False
)->tuple:
    
    local = os.path.join(
        base, stn, "individual", str(date.year), 
        str(date.timetuple().tm_yday), instr
    )
    if clean: shutil.rmtree(local, ignore_errors=True)
    os.makedirs(
        local, 
        exist_ok=True
    )
    logger.info(f"Created local folder: {local}")
    remote = local.replace("/tmp/", "/media/")
    logger.info(f"Remote folder: {remote}")
    return local, remote

def get_ngi_files_by_hour(date:dt.datetime, hours:list, remote:str)->list:

    remote_files = []
    for h in np.arange(hours[0], hours[1]+1):
        ftag = f"*{str(date.year)}{str(date.timetuple().tm_yday)}{'%02d'%h}*.ngi.bz2"
        re = os.path.join(
            remote, ftag
        )
        remote_files.extend(
            glob.glob(re)
        )
    remote_files.sort()
    logger.info(f"Total Files: {len(remote_files)}")
    return remote_files

def copy2local(local:str, remote_files:list):
    import os
    for remote_file in remote_files:
        fname = remote_file.split("/")[-1]
        shutil.copy2(remote_file, os.path.join(local, fname))
    return

def clean_based_on_labels(labels, traces):
    drop_list = []
    for l in labels:
        p = traces[traces.labels==l].copy()
        pf_max, pf_min, ph_max, ph_min = (
            p.frequency.max(), p.frequency.min(), p.height.max(), p.height.min()
        )
        qlabels = labels[labels!=l]
        for ql in qlabels:
            q = traces[traces.labels==ql].copy()
            qf_max, qf_min, qh_max, qh_min = (
                q.frequency.max(), q.frequency.min(), q.height.max(), q.height.min()
            )
            logger.info(f"{ql}, {qf_max}, {qf_min}, {qh_max}, {qh_min}")
            if (pf_max <= qf_max) and (ph_max>=qh_min) and (pf_min>=qf_min):
                logger.info(f"{ql}, {l}, {pf_max}, {qf_max}, {ph_max}, {qh_min}")
                drop_list.append(l)
                break
    select_list = np.array([l for l in labels if l not in drop_list])
    logger.info(f"{labels}, {drop_list}, {select_list}")
    return select_list