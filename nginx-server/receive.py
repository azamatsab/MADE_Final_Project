import av
import logging

logging.basicConfig(filename="rec.log",
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

logging.info("Running Streaming Service")

logger = logging.getLogger('receiver')

logger.info("Ready to open stream")

# Откроем ресурс на чтение
input_resource = av.open(
    'rtmp://localhost:1935/src/src'
)

logger.info("Stream opened")

# Список потоков входного ресурса: видео и аудио
input_streams = list()

# Список потоков выходного ресурса: видео и аудио
output_streams = list()

# Для входного и выходного ресурсов возьмём поток видео.
for stream in input_resource.streams:
    logger.info(stream.type)
    if stream.type == 'video':
        input_streams += [stream]
        break

# Для входного и выходного ресурсов возьмём поток аудио.
for stream in input_resource.streams:
    if stream.type == 'audio':
        input_streams += [stream]
        break

# В этом списке будем хранить пакеты выходного потока.
output_packets = list()

# Применим «инверсное мультиплексирование». Получим пакеты из потока.
for packet in input_resource.demux(input_streams):
    # Получим все кадры пакета.
    for frame in packet.decode():
        # Сбросим PTS для самостоятельного вычислении при кодировании.
        frame.pts = None
        #logger.info("Frame received")

