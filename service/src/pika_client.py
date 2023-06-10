import logging

import pika
from aio_pika import connect_robust


async def consume(loop, process_incoming_message):
    """Setup message listener with the current running loop"""
    connection = await connect_robust(host="rabbitmq",
                                    port=5672,
                                    loop=loop)
    channel = await connection.channel()
    queue = await channel.declare_queue("A")
    await queue.consume(process_incoming_message, no_ack=True)
    logging.info("Established pika async listener")
    return connection
