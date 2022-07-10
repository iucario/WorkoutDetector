import base64
import os
from pathlib import Path

import pandas as pd
import yt_dlp
from workoutdetector.settings import PROJ_ROOT


def parse_onedrive(link: str) -> str:
    """Parse onedrive link to download link.

    Args:
        link: str, start with `https://1drv.ms/u/s!`

    Returns:
        str, download link.
    """
    assert link.startswith('https://1drv.ms/u/s!')
    b = base64.urlsafe_b64encode(link.strip().encode('ascii'))
    s = b.decode('ascii')
    res = f'https://api.onedrive.com/v1.0/shares/u!{s}/root/content'
    return res


def download_ytb(url, folder='~'):
    vid = url[-11:]
    ydl_opts = {
        'outtmpl': f'{folder}/{vid}.mp4',
        'quiet': True,
        'ignoreerrors': True,
        'remux-video': 'mp4',
        'format': 'bv[height<=720]',  # 136 for mp4 1280x720 30fps no audio
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def download_repcount(csv_path, folder='~'):
    df = pd.read_csv(csv_path)
    for i, row in df.iterrows():
        vid = row['vid']
        st = row['st']
        if pd.isna(vid):
            continue
        url = f'https://www.youtube.com/watch?v={vid}'
        if os.path.exists(f'{folder}/{vid}.mp4'):
            print('skip', vid)
            continue
        download_ytb(url, folder)


def _get_url():
    URL_VIDEO = parse_onedrive('https://1drv.ms/u/s!AiohV3HRf-34ipk0i1y2P1txpKYXFw')
    URL_ANNO = parse_onedrive('https://1drv.ms/u/s!AiohV3HRf-34i_YvMob5Vpgvxjc3mQ')
    URL_RAWFRAME = parse_onedrive('https://1drv.ms/u/s!AiohV3HRf-34ipwACYfKSHhkZzebrQ')
    COUNTIX = parse_onedrive('https://1drv.ms/u/s!AiohV3HRf-34ipwWwzztzGynyj5Fwg')
    print(COUNTIX)


if __name__ == '__main__':
    # download_repcount(Path(PROJ_ROOT, 'datasets/RepCount/all_data.csv'),
    #                   '/mnt/d/repcount-redown')
    _get_url()
