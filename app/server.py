# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import json
import os
from typing import List, Set
import numpy as np

from fastapi import Body, FastAPI, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import tempfile
import io
from PIL import Image
from base64 import b64decode

from inference import  inference_video, get_frame



app = FastAPI()
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="my-app/build/static"), name="static")

@app.get("/")
@app.get("/index")
async def read_root():
    return FileResponse(path="my-app/build/index.html", media_type="text/html")

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket)
    print(f"Client {client_id} connected")
    try:
        while True:
            recv = await websocket.receive_text()
            if recv == 'close':
                await websocket.close()
                break
            elif recv == 'ping':
                await websocket.send_text('pong')
            else:
                try:
                    recv = recv.split(',')[1]  # base64
                    img = b64decode(recv)
                    image = np.array(Image.open(io.BytesIO(img)))
                    pred = await get_frame(image)
                    await manager.send_personal_message(json.dumps(pred))
                except Exception as e:
                    continue
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print(f'{websocket} disconnected')


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




# example_home = 'example_videos/'
# example_videos = [os.path.join(example_home, x)
#                   for x in os.listdir(example_home)]

# video = 'example_videos/2jpteC44QKg.mp4'
# main(video)
