import os
import pandas as pd
import argparse
import subprocess
import yaml

config = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '../utils/config.yml')))


def cut_video(label, vid, start_sec, end_sec, output_folder):
    input_file = os.path.join(config.proj_root, 'data', label, f'{vid}.mp4')
    output_file = os.path.join(output_folder, f'{vid}_{start_sec}.mp4')
    cmd = f'ffmpeg -i {input_file} -ss {start_sec} -to {end_sec}'\
        f'  -filter:v fps=30 {output_file}'
    subprocess.call(cmd, shell=True)
    # print(cmd)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-o', '--output', help='Path to output folder',
                      required=True)
    args = args.parse_args()
    df = pd.read_csv(os.path.join(config.proj_root, 'data/anno.csv'))
    df.apply(lambda row: cut_video(row['label'], row['vid'],
             row['start_sec'], row['end_sec'], output_folder=args.output), axis=1)
