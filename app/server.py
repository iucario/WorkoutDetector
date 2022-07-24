# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import io
import os
import tempfile
from base64 import b64decode
from collections import Counter
import time
from typing import Dict, List, Set

import numpy as np
from fastapi import Body, FastAPI, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision.io import VideoReader

from inference import count_rep_video, get_frame

sample_length = 8
app = FastAPI()
origins = [
    "http://localhost",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:

    def __init__(self):
        # Websocket: client_id
        self.active_connections: Dict[WebSocket, str] = dict()

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        print(f"Client {client_id} connected")
        self.active_connections[websocket] = client_id

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            client_id = self.active_connections.pop(websocket)
            print(f"Client {client_id} disconnected")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


class Stop(Exception):
    pass


async def receive(websocket: WebSocket, queue: asyncio.Queue):
    recv = await websocket.receive_text()
    if recv.startswith('data:image/webp;base64,'):
        recv = recv.split(',')[1]  # base64
        img = b64decode(recv)
        image = np.array(Image.open(io.BytesIO(img)))
        # print(f'got {image.shape} from client')
        await queue.put(image)
        # print(pred)
        # await manager.send_personal_message(json.dumps(pred), websocket)
    if recv == 'stop':
        raise Stop()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=16)
    detect_task = asyncio.create_task(get_frame(websocket, queue))
    try:
        while True:
            await receive(websocket, queue)
    except Stop:
        print('stopping')
        detect_task.cancel()
    except WebSocketDisconnect:
        detect_task.cancel()
    finally:
        manager.disconnect(websocket)


@app.post("/image")
async def read_image(payload: dict = Body(...)):
    return {'msg': 'Deprecated'}


@app.post("/video")
async def read_video(video: bytes = File(...)):
    if not video:
        return {'msg': 'No video'}
    video_path = os.path.join(tempfile.gettempdir(), 'tmp.mp4')
    with open(video_path, 'wb') as f:
        f.write(video)
    count, reps, actions = count_rep_video(video_path)
    if actions:
        action = Counter(actions).most_common(1)[0]
        return dict(success=True,
                    msg='success',
                    type='rep',
                    data={
                        'score': {
                            action[0]: 1.0
                        },
                        'count': {
                            action[0]: action[1]
                        },
                    })
    return dict(success=False, msg='no action')


def process_video(fname: str) -> None:
    if not os.path.exists(fname):
        print(f'{fname} not found')
        return
    vr = VideoReader(fname)
    meta = vr.get_metadata()
    print(meta)


@app.websocket("/test/{client_id}")
async def stream_test(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            recv = await websocket.receive_bytes()
            print(f'got len={len(recv)} from client')
            ts = time.time()
            with open(f'recv_{ts}.webm', 'wb') as f:
                f.write(recv)
            
    except Exception as e:
        print('[Exception]', e)
    finally:
        manager.disconnect(websocket)


# app.mount("/", StaticFiles(directory="my-app/build", html=True), name="static")
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app,
                host='localhost',
                port=8000,
                reload=True,
                debug=True,
                workers=1,
                ssl_keyfile=os.path.expanduser('~/cert/localhost+2-key.pem'),
                ssl_certfile=os.path.expanduser('~/cert/localhost+2.pem'))
