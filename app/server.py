# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from collections import deque
import json
import os
from time import sleep
from typing import Dict, List, Set
import numpy as np

from fastapi import Body, FastAPI, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import tempfile
import io
from PIL import Image
from base64 import b64decode

from inference import inference_video, get_frame

sample_length = 8
app = FastAPI()
origins = [
    "http://localhost",
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

        # ws: [state, frame_queue, result_queue]
        self.active_connections: Dict(WebSocket) = {}
        self.num_recv = 0
        self.num_pred = 0

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = [
            False, deque(maxlen=sample_length), deque(maxlen=1)]

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.pop(websocket)

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
        manager.num_recv += 1
        img = b64decode(recv)
        image = np.array(Image.open(io.BytesIO(img)))
        # print(f'got {image.shape} from client')
        await queue.put(image)
        manager.num_pred += 1
        # print(pred)
        # await manager.send_personal_message(json.dumps(pred), websocket)
    if recv == 'stop':
        raise Stop()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    print(f"Client {client_id} connected")
    queue = asyncio.Queue(maxsize=16)
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
        await manager.disconnect(websocket)


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
    pred = inference_video(video_path)
    return pred


app.mount("/", StaticFiles(directory="my-app/build", html=True), name="static")
