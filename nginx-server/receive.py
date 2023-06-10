import sys
import logging
import base64
import time
from io import BytesIO

import av
import requests
import pika


logging.basicConfig(filename="rec.log",
                            filemode="a",
                            format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)

logging.info("Running Streaming Service")

logger = logging.getLogger("receiver")
logger.info("Ready to open stream")

input_resource = av.open(
    "rtmp://localhost:1935/src/src"
)

logger.info("Stream opened")


class PikaClient:
    def __init__(self):
        credentials = pika.PlainCredentials("guest", "guest")
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host="rabbitmq", credentials=credentials)
        )
        self.channel = self.connection.channel()
        self.publish_queue = self.channel.queue_declare(queue="A")
        self.callback_queue = self.publish_queue.method.queue
        logger.info("Pika connection initialized")

    def send_message(self, message):
        """Method to publish message to RabbitMQ"""
        self.channel.basic_publish(
            exchange='',
            routing_key="A",
            body=message
        )

client = PikaClient()
input_streams = list()

for stream in input_resource.streams:
    logger.info(stream.type)
    if stream.type == "video":
        input_streams += [stream]
        break

for packet in input_resource.demux(input_streams):
    for frame in packet.decode():
        start = time.time()
        img = frame.to_image()

        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        # logger.error("Image time {}".format(time.time() - start))
        # requests.post("http://fastapi:5000/put", data=img_str)
        # logger.error("request {}".format(time.time() - start))
        logger.info("Writing ...")
        client.send_message(img_str)
        logger.info("Done ...")

