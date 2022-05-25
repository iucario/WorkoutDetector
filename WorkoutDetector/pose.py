import argparse
import math
import os
import os.path as osp

import decord
import mmcv
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

try:
    import mmdet
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')

try:
    import mmpose
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')

default_mmdet_root = '/home/umi/projects/mmdetection'
default_mmpose_root = '/home/umi/projects/mmpose'
default_tmpdir = './tmp'

default_det_config = (
    f'{default_mmdet_root}/configs/faster_rcnn/'
    'faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py')
default_det_ckpt = (
    'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
    'faster_rcnn_r50_fpn_1x_coco-person/'
    'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth')
default_pose_config = (
    f'{default_mmpose_root}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/'
    'coco/hrnet_w32_coco_256x192.py')
default_pose_ckpt = (
    'https://download.openmmlab.com/mmpose/top_down/hrnet/'
    'hrnet_w32_coco_256x192-c78dce93_20200708.pth')


def extract_frame(video_path):
    vid = decord.VideoReader(video_path)
    return [x.asnumpy() for x in vid]


def detection_inference(model, frames):
    results = []
    for frame in frames:
        result = inference_detector(model, frame)
        results.append(result)
    return results


def pose_inference(model, frames, det_results):
    assert len(frames) == len(det_results)
    total_frames = len(frames)
    num_person = max([len(x) for x in det_results])
    kp = np.zeros((num_person, total_frames, 17, 3), dtype=np.float32)

    for i, (f, d) in enumerate(zip(frames, det_results)):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        for j, item in enumerate(pose):
            kp[j, i] = item['keypoints']
    return kp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',  type=str, default='',
                        help='input video file')
    parser.add_argument('-l', '--video-list', type=str,
                        help='the list of source videos')
    parser.add_argument(
        '-b', '--base-dir', type=str,
        help='the base directory of the video list.'
        'file_path = base_dir/video_name')
    # * out should ends with '.pkl'
    parser.add_argument('-o', '--out', type=str, help='output pickle name')
    parser.add_argument('-t', '--tmpdir',  type=str, default=default_tmpdir)
    args = parser.parse_args()
    return args


def process_detection(det_results, threshold=0.5):
    # * Get detection results for human
    det_results = [x[0] for x in det_results]
    for i, res in enumerate(det_results):
        # * filter boxes with small scores
        res = res[res[:, 4] >= threshold]
        # * filter boxes with small areas
        box_areas = (res[:, 3] - res[:, 1]) * (res[:, 2] - res[:, 0])
        assert np.all(box_areas >= 0)
        res = res[box_areas >= threshold]
        det_results[i] = res
    return det_results


def inference_one_video(filepath, det_model, pose_model, tmpdir=None):
    if not os.path.exists(filepath):
        print(f'{filepath} does not exist. Skip.')
        return None
    # * Load video
    frames = extract_frame(filepath)
    # * Detect human
    det_results = detection_inference(det_model, frames)
    # * Detect human
    det_results = process_detection(det_results)
    # * Extract pose
    pose_results = pose_inference(pose_model, frames, det_results)
    # * Save to pickle
    if tmpdir:
        out_path = osp.join(tmpdir, osp.basename(filepath)+'.pkl')
        mmcv.dump(pose_results, out_path)
    return pose_results


def main():
    args = parse_args()
    pose_config = default_pose_config
    pose_ckpt = default_pose_ckpt
    det_config = default_det_config
    det_ckpt = default_det_ckpt

    if args.input:
        # Init detector
        det_model = init_detector(det_config, det_ckpt, 'cuda')
        assert det_model.CLASSES[0] == 'person', 'A detector trained on COCO is required'
        pose_model = init_pose_model(pose_config, pose_ckpt, 'cuda')
        pose_results = inference_one_video(
            args.input, det_model, pose_model, args.tmpdir)

    else:
        assert osp.exists(args.video_list), f'{args.video_list} not exists'
        assert osp.exists(args.base_dir), f'{args.base_dir} not exists'
        assert osp.exists(args.out), f'{args.out} not exists'
        os.makedirs(args.tmpdir, exist_ok=True)

        # Read video list
        with open(args.video_list, 'r') as f:
            video_list = f.read().splitlines()
        video_list = [x.split() for x in video_list]
        labels = []
        if len(video_list[0]) == 1:
            video_list = [osp.join(args.base_dir, x) for x in video_list]
        elif len(video_list[0]) == 2:
            labels = [x[1] for x in video_list]

        det_model = init_detector(det_config, det_ckpt, 'cuda')
        assert det_model.CLASSES[0] == 'person', 'A detector trained on COCO is required'
        pose_model = init_pose_model(pose_config, pose_ckpt, 'cuda')

        # Inference
        for idx, filename in enumerate(tqdm(video_list)):
            if len(labels) > 0:
                label = labels[idx]
            else:
                label = filename
            pose_results = inference_one_video(
                filename, det_model, pose_model, args.tmpdir)

    print(pose_results)


if __name__ == '__main__':
    main()
