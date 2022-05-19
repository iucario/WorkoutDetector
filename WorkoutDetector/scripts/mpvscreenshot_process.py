import os
import csv
import argparse
from config import BASEDIR

test = {
    'mpv-3VcKaXpzqRo.mp4-00_00_35.png': '3VcKaXpzqRo_00_00_35.png',
    'mpv-3g-1J2KkX_8.mp4-00_00_10.png': '3g-1J2KkX_8_00_00_10.png',
}

parser = argparse.ArgumentParser(description='Process mpv screenshots')
parser.add_argument('-i', '--input', help='Path to mpv screenshots', required=True)
parser.add_argument('-o', '--output', help='Path to output csv', required=False)


def process(path):
    for file in os.listdir(path):
        if file.endswith('.png'):
            new_file = file.replace('mpv-', '').replace('.mp4', '')
            new_file = new_file[:11] + '_' + new_file[12:]
            os.rename(os.path.join(path, file), os.path.join(path, new_file))
            print(new_file)


def to_csv(path) -> "list[str]":
    annos = []
    label = os.path.split(path)[-1]
    for file in os.listdir(path):
        if file.endswith('.png'):
            vid = file[:11]
            start_time = file[12:20].replace('_', ':')
            start_sec = start_time.split(':')
            start_sec = int(start_sec[0]) * 3600 + \
                int(start_sec[1]) * 60 + int(start_sec[2])
            end_sec = start_sec + 5
            annos.append([vid, label, start_sec, end_sec])
            print(f'{vid}, {label}, {start_sec}, {end_sec}')
    return annos


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.output:
        args.output = os.path.join(BASEDIR, 'data/anno.csv')
    annos = to_csv(args.input)
    if args.output:
        with open(args.output, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(annos)
