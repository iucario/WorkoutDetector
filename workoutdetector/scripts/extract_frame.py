import os
import cv2


def extract(video: str, dest_dir: str):
    """Extract frames from one video to `dest_dir/{video}/img_{idx}.jpg`."""

    name_no_ext = os.path.basename(video).split('.')[0]
    out_dir = os.path.join(dest_dir, name_no_ext)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    idx = 1
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(out_dir, f'img_{idx:05}.jpg')
        cv2.imwrite(out_path, frame)
        idx += 1
    cap.release()
    print(f'{video} done')


def extract_all(data_dir: str, dest_dir: str):
    """Extract frames from all videos in `data_dir` to `dest_dir`."""

    for split in os.listdir(data_dir):
        for video in os.listdir(os.path.join(data_dir, split)):
            extract(os.path.join(data_dir, split, video), os.path.join(dest_dir, split))

if __name__ == '__main__':
    data_root = os.path.expanduser('~/data/RepCount/videos/')
    dest_dir = os.path.expanduser('~/data/RepCount/rawframes/')
    extract_all(data_root, dest_dir)
    # for video in os.listdir(data_root):
    #     video_path = os.path.join(data_root, video)
    #     extract(video_path, dest_dir)