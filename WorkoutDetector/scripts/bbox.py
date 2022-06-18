from WorkoutDetector.settings import PROJ_ROOT, REPCOUNT_ANNO_PATH
from WorkoutDetector.datasets import RepcountHelper
import os
import numpy as np
import cv2
import torch
from typing import List
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import PIL

data_transform = T.Compose([
    T.ToTensor(),
    # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def crop_and_save(results, frames, out_path, fps) -> None:
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


def draw_person_boxes(results, frames, out_path, width, height, fps) -> None:
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for idx, frame in enumerate(frames):
        result = results[idx]
        person_boxes = result[0]['boxes'][result[0]['labels'] == 1]
        scores = result[0]['scores'][result[0]['labels'] == 1]
        person_boxes = person_boxes[scores > 0.7]
        t = torch.from_numpy(np.array(frame)).permute(2, 0, 1)
        drawn = draw_bounding_boxes(t,
                                    person_boxes[:4],
                                    colors=['red', 'green', 'yellow', 'orange'],
                                    width=2)
        drawn = TF.to_pil_image(drawn)
        out.write(np.array(drawn, dtype=np.uint8)[:, :, ::-1])
    out.release()


if __name__ == '__main__':
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.cuda()
    helper = RepcountHelper(os.path.join(PROJ_ROOT, 'data/RepCount'), REPCOUNT_ANNO_PATH)
    data_dict: dict = helper.get_rep_data(split=['val'], action=['push_up'])

    for item in data_dict.values():
        video_path = item.video_path
        print(video_path)
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
            imgs = [data_transform(f) for f in frames]
            results = [model(img.unsqueeze(0).cuda()) for img in imgs]

        draw_person_boxes(results, frames, out_path, width, height, fps)