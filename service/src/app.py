import sys
import shutil
import logging
import time
import base64
import asyncio
from io import BytesIO
from threading import Thread

import uvicorn
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request, Header, Response
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
from PIL import Image
import pika

from processor import Processor
from reader import read_video, Reader, Writer
from tools import id_generator, zipfiles
from configs import BUFFER_SIZE, RESTART_ITER
from pika_client import consume

processor = Processor("cpu")

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


credentials = pika.PlainCredentials("guest", "guest")

# def startConsumer():
#     while True:
#         try:
#             connection = pika.BlockingConnection(pika.ConnectionParameters(host="rabbitmq", port="5672", credentials=credentials))
#             channel = connection.channel()
#             channel.exchange_declare("test", durable=True, exchange_type="topic")
#             channel.basic_consume(queue="A", on_message_callback=callbackFunctionForQueueA, auto_ack=True)
#             consumer_thread = Thread(target=channel.start_consuming, daemon=True)
#             consumer_thread.start()
#         except:
#             time.sleep(0.001)

@app.on_event("startup")
async def startup():
    # loop = asyncio.get_running_loop()
    loop = asyncio.get_event_loop()
    # loop.run_until_complete(asyncio.wait(futures))
    task = loop.create_task(consume(loop, callbackFunctionForQueueA))
    await task

def callbackFunctionForQueueA(message):
    body = message.body
    if body:
        img = Image.open(BytesIO(base64.b64decode(body)))
        open_cv_image = np.array(img) 
        open_cv_image = open_cv_image[:, :, ::-1]
        frame_reader.put(open_cv_image, "stream", time.time())

@app.get("/play", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("stream_video.html", context={"request": request})


@app.get("/download_vgif")
async def download_videogif(request: Request):
    filenames = frame_reader.video_info.get_gif()
    filenames2 = frame_reader.video_info.get_data()
    stream, zip_filename = zipfiles(filenames + filenames2)
    return Response(stream.getvalue(), media_type="application/x-zip-compressed", headers={
        'Content-Disposition': f'attachment;filename={zip_filename}'
    })

@app.get("/download_sgif")
async def download_videogif(request: Request):
    filenames = frame_reader.stream_info.get_gif()
    filenames2 = frame_reader.stream_info.get_data()
    stream, zip_filename = zipfiles(filenames + filenames2)
    return Response(stream.getvalue(), media_type="application/x-zip-compressed", headers={
        'Content-Disposition': f'attachment;filename={zip_filename}'
    })

def streamer():
    restart = RESTART_ITER
    while restart > 0:
        try:
            while not frame_reader.out_video.empty():
                if restart != RESTART_ITER:
                    restart = RESTART_ITER
                frame = frame_reader.out_video.get()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, encodedImage = cv2.imencode(".jpg", frame)
                yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n\r\n')
        except GeneratorExit:
            logging.info("Cancelling generator")
        restart -= 1
        time.sleep(0.05)

def streamer2():
    restart = RESTART_ITER
    while restart > 0:
        try:
            while frame_reader.out_stream.qsize() >= BUFFER_SIZE:
                if restart != RESTART_ITER:
                    restart = RESTART_ITER
                frame = frame_reader.out_stream.get()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, encodedImage = cv2.imencode(".jpg", frame)
                yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n\r\n')
        except GeneratorExit:
            logging.info("Cancelling generator")
        restart -= 1
        time.sleep(0.05)

@app.get("/video")
def video_endpoint(request: Request):
    return StreamingResponse(streamer(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stream")
def stream_endpoint():
    return StreamingResponse(streamer2(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/", response_class=HTMLResponse)
async def get_main(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def post_main(request: Request, file: UploadFile = File(...)):
    filename = id_generator() + ".mp4"
    with open(filename, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    cap = cv2.VideoCapture(filename)
    writer = Writer(cap, frame_reader)
    writer.setDaemon(True)
    writer.start()
    time.sleep(1)
    return templates.TemplateResponse("img_video.html", context={"request": request})

@app.post("/put", response_class=HTMLResponse)
async def rtmp_frame(request: Request):
    contents = await request.body()
    img = Image.open(BytesIO(base64.b64decode(contents)))
    open_cv_image = np.array(img) 
    open_cv_image = open_cv_image[:, :, ::-1]
    frame_reader.put(open_cv_image, "stream", time.time())
    return JSONResponse([{"status": "ok"}])

if __name__ == "__main__":
    frame_reader = Reader(processor)
    frame_reader.setDaemon(True)
    frame_reader.start()
    uvicorn.run(app, host="0.0.0.0", port=5000)
