import json
import os
from typing import List

import cv2
import numpy as np
import PIL
import torch
from torch import Tensor
from torchvision.io import read_video
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.utils import draw_bounding_boxes
from workoutdetector.datasets import RepcountHelper
from workoutdetector.settings import PROJ_ROOT, REPCOUNT_ANNO_PATH

data_transform = T.Compose([
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def crop_and_save(results: List[List[dict]], frames: List[np.ndarray], out_path: str,
                  fps: int) -> None:
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (224, 224))
    for idx, frame in enumerate(frames):
        result = results[idx]
        person_boxes = result[0]['boxes'][result[0]['labels'] == 1]
        scores = result[0]['scores'][result[0]['labels'] == 1]
        person_boxes = person_boxes[scores > 0.7]
        # t = torch.from_numpy(np.array(frame)).permute(2, 0, 1)
        t = frame
        # drawn = draw_bounding_boxes(t, person_boxes, colors='red')
        if len(person_boxes) > 0:
            x1, y1, x2, y2 = person_boxes[0].cpu().numpy()
            print(x1, y1, x2, y2)
            t = TF.crop(t, y1, x1, y2 - y1, x2 - x1)
        r = TF.resize(t, (224, 224))
        # drawn.save(f'tmp/bbox_{idx}.png')
        out.write(np.array(r, dtype=np.uint8)[:, :, ::-1])
    out.release()


def draw_person_boxes(results:List[dict], frames: List[np.ndarray], out_path: str,
                      width: int, height: int, fps: int) -> None:
    """
    Args:
        results: List[dict], same length as frames. Item is detector output for each frame.
        frames: List[np.ndarray], list of PIL Image.
        out_path: str, path to output drawn video.
        width: int, width of output video.
        height: int, height of output video.
        fps: int, fps of output video.
    """
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for idx, frame in enumerate(frames):
        result = results[idx]
        person_boxes: List[int] = result['boxes']
        t = torch.from_numpy(np.array(frame)).permute(2, 0, 1)
        drawn = draw_bounding_boxes(t,
                                    torch.tensor(person_boxes[:4]),
                                    colors=['red', 'green', 'yellow', 'orange'],
                                    width=2)
        drawn = TF.to_pil_image(drawn)
        out.write(np.array(drawn, dtype=np.uint8)[:, :, ::-1])
    out.release()


def bboxes_to_json(results: List[List[dict]], out_path: str) -> None:
    # TODO: handle no person
    assert out_path.endswith('.json')
    threshold = 0.7
    with open(out_path, 'w') as f:
        l = []
        for result in results:
            d = dict()
            person_boxes = result[0]['boxes'][result[0]['labels'] == 1]
            scores = result[0]['scores'][result[0]['labels'] == 1]
            scores = scores[scores > threshold]
            person_boxes = person_boxes[scores > threshold]
            person_boxes.cpu().numpy()
            d['boxes'] = person_boxes.tolist()
            d['scores'] = scores.tolist()
            l.append(d)
        json.dump(l, f)


def json_to_bboxes(json_path: str) -> List[dict]:
    with open(json_path, 'r') as f:
        l = json.load(f)
    return l


def _video_to_json(model: torch.nn.Module):
    """Detect person in videos and save to json files for each video."""

    helper = RepcountHelper(os.path.join(PROJ_ROOT, 'data/RepCount'), REPCOUNT_ANNO_PATH)
    data_dict: dict = helper.get_rep_data(
        split=['test'],
        action=['situp', 'push_up', 'pull_up', 'jump_jack', 'squat', 'front_raise'])

    for item in data_dict.values():
        video_path = item.video_path
        print(video_path)
        json_out_path = os.path.join(PROJ_ROOT, f'out/{item.video_name}.json')
        if os.path.isfile(json_out_path):
            print('skip')
            continue
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(PIL.Image.fromarray(frame))
        cap.release()

        out_path = os.path.join(PROJ_ROOT, f'out/bbox-{item.video_name}')
        with torch.no_grad():
            # imgs = [data_transform(f) for f in frames]
            results = [model(TF.to_tensor(img).unsqueeze(0).cuda()) for img in frames]
            bboxes_to_json(results, json_out_path)

def process_bbox(results: List[List[dict]], frames: List[np.ndarray], out_path: str) -> None:
    """Post process bounding boxes.
    Select the largest bbox and crop the image and resize to 224*224.

    Args:
        results: List[dict], batch of detector output.
        frames: List[np.ndarray], list of PIL Image.
        out_path: str, path to output cropped and resized video.

    Algorithm:
    1. Select the largest bbox.
    2. Crop the image and resize to 224*224.
    """
    #TODO: input 8 frames, make the bounding boxes consistent.
    #TODO: select the center bbox if there are multiple bboxes.

if __name__ == '__main__':
    # model = fasterrcnn_resnet50_fpn(pretrained=True)
    # model.eval()
    # model.cuda()
    json_files = [
        j for j in os.listdir(os.path.join(PROJ_ROOT, 'out')) if j.endswith('.json')
    ]
    for j in json_files:
        results = json_to_bboxes(os.path.join(PROJ_ROOT, 'out', j))
        frames, _, metadata = read_video(os.path.join(PROJ_ROOT, 'data/RepCount/videos/test', j.replace('.json', '')))
        fps = metadata['video_fps']
        height, width = frames[0].shape[:2]
        print(width, height, fps)
        out_path = os.path.join(PROJ_ROOT, f'out/bbox-{j.replace(".json", "")}')
        # print(results[0])
        draw_person_boxes(results, frames, out_path, width, height, fps)
