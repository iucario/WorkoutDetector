import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from WorkoutDetector.datasets import RepcountDataset, RepcountImageDataset
import onnx
import onnxruntime


def inference_video(ort_session, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    input_name = ort_session.get_inputs()[0].name
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig_frame = frame.copy()
        frame = cv2.resize(frame, (224, 224))
        frame = np.transpose(frame, (2, 0, 1))
        frame = np.expand_dims(frame, axis=0)
        frame = frame.astype(np.float32)
        ort_inputs = {input_name: frame}
        ort_outs = ort_session.run(None, ort_inputs)
        score = ort_outs[0][0]
        pred = labels[score.argmax()]
        text = pred + ': ' + str(round(max(score), 2))
        if pred == 'down':
            color = (0, 0, 255)
        elif pred == 'up':
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)
        cv2.putText(orig_frame, text, (int(width*0.3), int(height*0.3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        out.write(orig_frame)
    cap.release()
    out.release()


if __name__ == '__main__':
    data_root = '/home/umi/projects/WorkoutDetector/data'
    dataset = RepcountDataset(root=data_root, split='test')
    action = dataset.df[dataset.df['class'] == 'squat']
    l = action['name'].values
    print(l)

    ort_session = onnxruntime.InferenceSession(
        '/home/umi/projects/WorkoutDetector/WorkoutDetector/lightning_logs/version_1/checkpoints/epoch=207-step=14560.onnx'
    )

    labels = ['down', 'up']
    rand_video = os.path.join(data_root, 'RepCount/videos/test', l[0])
    print(rand_video)
    inference_video(ort_session, rand_video, 'tmp.mp4')
