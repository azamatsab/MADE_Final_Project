import sys
import shutil

import uvicorn
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request, Header, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import imageio
import numpy as np

from processor import Processor
from reader import read_video, Reader


processor = Processor("cpu")

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

CHUNK_SIZE = 1024 * 1024
video_path = Path("test.mp4")


@app.get("/play", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("video.html", context={"request": request})


@app.get("/video")
async def video_endpoint(range: str = Header(None)):
    start, end = range.replace("bytes=", "").split("-")
    start = int(start)
    end = int(end) if end else start + CHUNK_SIZE
    with open(video_path, "rb") as video:
        video.seek(start)
        data = video.read(end - start)
        filesize = str(video_path.stat().st_size)
        headers = {
            'Content-Range': f'bytes {str(start)}-{str(end)}/{filesize}',
            'Accept-Ranges': 'bytes'
        }
        print(end, start)
        return Response(data, status_code=206, headers=headers, media_type="video/mp4")


@app.get("/", response_class=HTMLResponse)
async def get_main(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def post_main(request: Request, file: UploadFile = File(...)):
    filename = 'test.mp4'
    with open(filename, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    frames = read_video(filename, processor, 500)
    index = [i for i in range(20, 80, 5)]
    gif_path = "gif.gif"
    imageio.mimsave(gif_path, [face for face in np.array(frames)[index]], duration=0.5)
    sys.stdout.flush()
    return FileResponse(gif_path)


if __name__ == "__main__":
    frame_reader = Reader()
    frame_reader.setDaemon(True)
    frame_reader.start()

    uvicorn.run(app, host="0.0.0.0", port=5000)
