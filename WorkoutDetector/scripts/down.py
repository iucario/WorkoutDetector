import os
from pathlib import Path
import yt_dlp

HOME = Path.home()


def download_ytb(url, folder=os.path.join(HOME, 'tmp')):
    vid = url[-11:]
    ydl_opts = {
        'outtmpl': f'{folder}/{vid}.mp4',
        'quiet': True,
        'ignoreerrors': True,
        'remux-video': 'mp4',
        'format': '136',  # 136 for mp4 1280x720 30fps no audio
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


if __name__ == '__main__':
    while True:
        x = input('Youtube link or Q to quit: ')
        if x == 'q' or x == 'Q':
            break
        download_ytb(x)
