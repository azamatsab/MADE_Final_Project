import logging
import base64

import av
import requests
from PIL import Image
from io import BytesIO


logging.basicConfig(filename="rec.log",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

logging.info("Running Streaming Service")

logger = logging.getLogger('receiver')

logger.info("Ready to open stream")

input_resource = av.open(
    'rtmp://localhost:1935/src/src'
)

logger.info("Stream opened")

input_streams = list()

for stream in input_resource.streams:
    logger.info(stream.type)
    if stream.type == 'video':
        input_streams += [stream]
        break

for packet in input_resource.demux(input_streams):
    # Получим все кадры пакета.
    for frame in packet.decode():
        img = frame.to_image()

        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())

        requests.post("http://fastapi:5000/put", data=img_str)
