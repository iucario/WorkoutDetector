from bisect import bisect_left
from collections import deque
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from WorkoutDetector.datasets import RepcountDataset, RepcountImageDataset
import onnx
import onnxruntime

data_transform = T.Compose([
    T.ToPILImage(),
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def inference_image(ort_session, frame) -> int:
    # frame = cv2.resize(frame, (224, 224))
    # frame = np.transpose(frame, (2, 0, 1))
    # frame = np.expand_dims(frame, axis=0)
    # frame = frame.astype(np.float32)
    frame = data_transform(frame).unsqueeze(0).numpy()
    input_name = ort_session.get_inputs()[0].name
    ort_inputs = {input_name: frame}
    ort_outs = ort_session.run(None, ort_inputs)
    score = ort_outs[0][0]
    pred = score.argmax()

    return pred


def inference_video(ort_session, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pred = inference_image(ort_session, frame)
        text = labels[pred]
        if pred == 1:
            color = (0, 0, 255)
        elif pred == 0:
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.putText(frame, text, (int(width * 0.3), int(height * 0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        out.write(frame)
    cap.release()
    out.release()


def evaluate_video(ort_session, video_path, output_path, ground_truth: list):
    """Evaluate repetition count on a video, using image classification model."""

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (width, height))
    result = deque(maxlen=7)
    count = 0
    states = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        frame_idx += 1
        if not ret:
            break
        curr_pred = inference_image(ort_session, frame)
        result.append(curr_pred)
        pred = sum(result) > len(result)//2  # vote of frames
        if not states:
            states.append(pred)
        elif states[-1] != pred:
            states.append(pred)
            if pred != states[0]:
                count += 1
        text = f'{labels[curr_pred]}'
        if pred == 1:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.putText(frame, text, (int(width * 0.2), int(height * 0.2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f'pred {count}', (int(width * 0.2), int(height * 0.4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (52, 235, 177), 2)
        gt_count = bisect_left(ground_truth[1::2], frame_idx)
        cv2.putText(frame, f'true {gt_count}', (int(width * 0.2), int(height * 0.6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 235, 52), 2)
        out.write(frame)
    if abs(count - len(ground_truth)//2) <= 1:
        print(f'{video_path} Prediction is correct!')
    cap.release()
    out.release()


if __name__ == '__main__':
    data_root = '/home/umi/projects/WorkoutDetector/data'
    dataset = RepcountDataset(root=data_root, split='test')
    action = dataset.df[dataset.df['class'] == 'jump_jack']
    l = action['name'].values
    print(l)

    ort_session = onnxruntime.InferenceSession(
        '/home/umi/projects/WorkoutDetector/checkpoints/jump_jack_1x3x224x224_20220610.onnx'
    )

    labels = ['start', 'mid']
    for i in range(len(l)):
        rand_video = os.path.join(data_root, 'RepCount/videos/test', l[i])
        print(rand_video)
        gt = list(map(int, action['reps'].values[i].split(' ')))
        evaluate_video(ort_session, rand_video, f'/mnt/d/infer/{l[i]}', gt)
