import os
from pathlib import Path
import pandas as pd
import yt_dlp
from workoutdetector.settings import PROJ_ROOT


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


if __name__ == '__main__':
    download_repcount(Path(PROJ_ROOT, 'datasets/RepCount/all_data.csv'),
                      '/mnt/d/repcount-redown')