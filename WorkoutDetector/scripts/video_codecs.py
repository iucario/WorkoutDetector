import cv2
import tempfile


with tempfile.NamedTemporaryFile(suffix='.mp4') as f:
    print(f.name)
    out = cv2.VideoWriter(f.name, cv2.VideoWriter_fourcc(*'vp80'), 30, (640, 480))
    out.release()