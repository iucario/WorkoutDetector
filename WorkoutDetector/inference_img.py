import onnx
import onnxruntime
import numpy as np
import torch
import cv2

ort_session = onnxruntime.InferenceSession('lightning_logs/version_2/squat.onnx')
input_name = ort_session.get_inputs()[0].name
labels = ['down', 'up']


def inference_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

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
        cv2.putText(orig_frame, f'{pred} {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out.write(orig_frame)
    out.release()


inference_video('data/RepCount/videos/test/val1257.mp4', 'tmp.mp4')
