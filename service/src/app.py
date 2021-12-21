import sys
import shutil

import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import imageio
import numpy as np

from processor import Processor
from reader import read_video


processor = Processor("cpu")

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/basic", response_class=HTMLResponse)
async def get_basic(request: Request):
    return templates.TemplateResponse("item.html", {"request": request})

@app.post("/basic", response_class=HTMLResponse)
async def post_basic(request: Request, file: UploadFile = File(...)):
    filename = 'test.mp4'
    with open(filename, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    frames = read_video(filename, processor, 500)
    index = [i for i in range(20, 80, 5)]
    gif_path = "gif.gif"
    imageio.mimsave(gif_path, [face for face in np.array(frames)[index]], duration=0.5)
    print("Done")
    sys.stdout.flush()
    return FileResponse(gif_path)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
