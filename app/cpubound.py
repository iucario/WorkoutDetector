import asyncio
from collections import deque
import json
from math import inf
import os
from typing import Dict, List, Set, Tuple
import cv2
from cv2 import CascadeClassifier
import numpy as np

import uvicorn
from fastapi import Body, FastAPI, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tempfile
import io
from PIL import Image
from base64 import b64decode

from time import sleep

sample_length = 5
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

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
            <button onclick="sendStart()">Start</button>
            <button onclick="sendStop()">Stop</button>
        
        <ul id='send'>
         send
        </ul>
        <ul id='recv'>
        recv
        </ul>
        <script>
            let ws;
            let intervalId = null;
            let counter = 0;
            
            function sendStart(event) {
                ws = new WebSocket("ws://localhost:8000/test");
                intervalId = setInterval(() => {
                    ws.send(counter.toString())
                    var messages = document.getElementById('send')
                    var message = document.createElement('li')
                    var content = document.createTextNode(counter.toString())
                    message.appendChild(content)
                    messages.appendChild(message)
                    counter += 1;
                    }, 100);
                ws.onmessage = function(event) {
                    var messages = document.getElementById('recv')
                    var message = document.createElement('li')
                    var content = document.createTextNode(event.data)
                    message.appendChild(content)
                    messages.appendChild(message)
                };
            
            }
            function sendStop(event) {
                window.clearInterval(intervalId);
                ws.send('stop')
                ws.close();
            }
        </script>
    </body>
</html>
"""


class Stop(Exception):
    pass


async def receive(websocket: WebSocket, queue: asyncio.Queue):
    bytes = await websocket.receive_text()
    if bytes == 'stop':
        raise Stop()

    try:
        queue.put_nowait(bytes)
    except asyncio.QueueFull:
        pass


async def detect(websocket: WebSocket, queue: asyncio.Queue):
    que = []
    while True:
        data = await queue.get()
        que.append(data)
        print(f'got {data} from queue')
        if len(que) == sample_length:
            sleep(1)
            msg = f'range {que[0]}, {que[-1]}'
            print(f'received bytes of range {que[0]}, {que[-1]} from client')
            que.clear()
            await websocket.send_text(msg)


@app.websocket("/test")
async def face_detection(websocket: WebSocket):
    await websocket.accept()
    queue = asyncio.Queue(maxsize=5)
    detect_task = asyncio.create_task(detect(websocket, queue))
    try:
        while True:
            await receive(websocket, queue)
    except Stop:
        print('stopping')
        detect_task.cancel()

    except WebSocketDisconnect:
        detect_task.cancel()
        await websocket.close()


@app.get("/")
async def get():
    return HTMLResponse(html)


global status
status = False


async def cpu_bound(input_que: deque, result_que: deque):
    global status
    if status:
        sleep(1)
        if input_que:
            return str(input_que.popleft())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global status
    input_que = deque(maxlen=4)
    result_que = deque(maxlen=1)
    await websocket.accept()
    try:
        while True:
            recv = await websocket.receive_text()  # Fix: blocking
            if recv == 'close':
                await websocket.close()
                input_que.clear()
                result_que.clear()
                break
            elif recv == 'stop':
                status = False

                print(f"Client  stopped.\nTotal recv {num_recv}, pred {num_pred}")
                num_recv = 0
                num_pred = 0
                await websocket.send_text('Stopped')
            elif recv == 'start':
                status = True
                print(f"Client started")
            else:
                if status:
                    num_recv += 1
                    input_que.append(recv)
                    pred = await cpu_bound(input_que, result_que)
                    num_pred += 1
                    print(pred)
                    # is this IO bound?
                    await websocket.send_text(pred)

    except WebSocketDisconnect:
        websocket.close()
        print(f'{websocket} disconnected')


'''https://stackoverflow.com/questions/67947099/send-receive-in-parallel-using-websockets-in-python-fastapi'''


async def read_and_send_to_client(websocket, data):
    print(f'reading {data} from client')
    await asyncio.sleep(2)  # simulate a slow call
    print(f'finished reading {data}, sending to websocket client')
    await websocket.send_text(data)


@app.websocket("/wsqueue")
async def read_webscoket(websocket: WebSocket):
    await websocket.accept()
    queue = asyncio.queues.Queue(maxsize=5)

    async def read_from_socket(websocket: WebSocket):
        async for data in websocket.iter_text():
            print(f"putting {data} in the queue")
            queue.put_nowait(data)

    async def get_data_and_send():
        data = await queue.get()
        fetch_task = asyncio.create_task(read_and_send_to_client(websocket, data))
        while True:
            data = await queue.get()
            if data == 'stop':
                print('stopping')
                await websocket.send_text('Stopped')
                fetch_task.cancel()
                break
            else:
                fetch_task = asyncio.create_task(read_and_send_to_client(websocket, data))

    await asyncio.gather(read_from_socket(websocket), get_data_and_send())
