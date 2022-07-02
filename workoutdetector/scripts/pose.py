import argparse
import math
import os
import os.path as osp
from typing import List, Optional

import decord
import mmcv
import numpy as np
from tqdm import tqdm
import mmdet
from mmdet.apis import inference_detector, init_detector
import mmpose
from mmpose.apis import inference_top_down_pose_model, init_pose_model
import warnings

warnings.filterwarnings("ignore")

default_mmdet_root = '/workspace/mmdetection'
default_mmpose_root = '/workspace/mmpose'
default_tmpdir = '/workspace/tmp'

default_det_config = (f'{default_mmdet_root}/configs/faster_rcnn/'
                      'faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py')
default_det_ckpt = ('https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                    'faster_rcnn_r50_fpn_1x_coco-person/'
                    'faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth')
default_pose_config = (
    f'{default_mmpose_root}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/'
    'coco/hrnet_w32_coco_256x192.py')
default_pose_ckpt = ('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                     'hrnet_w32_coco_256x192-c78dce93_20200708.pth')


def extract_frame(video_path: str) -> List[np.ndarray]:
    vid = decord.VideoReader(video_path)
    return [x.asnumpy() for x in vid]


def detection_inference(model, frames) -> list:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input video file')
    parser.add_argument('-l', '--video-list', type=str, help='the list of source videos')
    parser.add_argument('-b',
                        '--base-dir',
                        type=str,
                        help='the base directory of the video list.'
                        'file_path = base_dir/video_name')
    # * out should ends with '.pkl'
    parser.add_argument('-o', '--out', type=str, help='output pickle name')
    parser.add_argument('-t', '--tmpdir', type=str, default=default_tmpdir)
    args = parser.parse_args()
    return args


def process_detection(det_results, threshold: float = 0.5):
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


def inference_one_video(filepath: str, det_model, pose_model) -> Optional[dict]:
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
    shape = frames[0].shape[:2]
    anno = dict(
        img_shape=shape,
        total_frames=len(frames),
        num_person=pose_results.shape[0],
        keypoint=pose_results[..., :2].astype(np.float16),
        keypoint_score=pose_results[..., 2].astype(np.float16),
    )

    return anno


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
        anno = inference_one_video(args.input, det_model, pose_model, args.tmpdir)
        print(anno)

    else:
        assert osp.exists(args.video_list), f'{args.video_list} not exists'
        assert osp.exists(args.base_dir), f'{args.base_dir} not exists'
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
        for idx, line in enumerate(tqdm(video_list)):
            filepath = osp.join(args.base_dir, line[0])
            anno = inference_one_video(filepath, det_model, pose_model)
            if args.tmpdir:
                out_path = osp.join(args.tmpdir, osp.basename(filepath) + '.pkl')
                mmcv.dump(anno, out_path)


if __name__ == '__main__':
    main()
