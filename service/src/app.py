import sys
import shutil
import logging
import time
from io import BytesIO
import base64

import uvicorn
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request, Header, Response
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import imageio
import numpy as np
import cv2
from PIL import Image

from processor import Processor
from reader import read_video, Reader, Writer
from tools import id_generator
from configs import BUFFER_SIZE

processor = Processor("cpu")

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

CHUNK_SIZE = 1024 * 1024
video_path = Path("test.mp4")


@app.get("/play", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("stream_video.html", context={"request": request})


# @app.get("/video")
# async def video_endpoint(range: str = Header(None)):
#     start, end = range.replace("bytes=", "").split("-")
#     start = int(start)
#     end = int(end) if end else start + CHUNK_SIZE
#     with open(video_path, "rb") as video:
#         video.seek(start)
#         data = video.read(end - start)
#         filesize = str(video_path.stat().st_size)
#         headers = {
#             'Content-Range': f'bytes {str(start)}-{str(end)}/{filesize}',
#             'Accept-Ranges': 'bytes'
#         }
#         return Response(data, status_code=206, headers=headers, media_type="video/mp4")


def streamer():
    try:
        while not frame_reader.out_video.empty():
            frame = frame_reader.out_video.get()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, encodedImage = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n\r\n')
    except GeneratorExit:
        logging.info("Cancelling generator")

def streamer2():
    try:
        while frame_reader.out_stream.qsize() >= BUFFER_SIZE:
            frame = frame_reader.out_stream.get()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, encodedImage = cv2.imencode(".jpg", frame)
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n\r\n')
    except GeneratorExit:
        logging.info("Cancelling generator")

@app.get("/video")
def video_endpoint(request: Request):
    return StreamingResponse(streamer(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stream")
def stream_endpoint():
    return StreamingResponse(streamer2(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/", response_class=HTMLResponse)
async def get_main(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})


# @app.post("/", response_class=HTMLResponse)
# async def post_main(request: Request, file: UploadFile = File(...)):
#     filename = id_generator() + ".mp4"
#     with open(filename, 'wb') as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     frames = read_video(filename, processor, 500)
#     index = [i for i in range(20, 80, 5)]
#     gif_path = id_generator() + ".gif"
#     imageio.mimsave(gif_path, [face for face in np.array(frames)[index]], duration=0.5)
#     return FileResponse(gif_path)


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
    frame_reader.put(open_cv_image, "stream")
    return JSONResponse([{"status": "ok"}])

if __name__ == "__main__":
    frame_reader = Reader(processor)
    frame_reader.setDaemon(True)
    frame_reader.start()

    uvicorn.run(app, host="0.0.0.0", port=5000)
