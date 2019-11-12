import logging
import os
import queue
import subprocess as sp

import numpy as np

from util.threadpoolutils import submit

logger = logging.getLogger("app")
DEBUG = False


class SourceQueue:
    def __init__(self, sources_parsed, max_queue_size=5):
        procs, queues, shape = [], [], []
        for source in sources_parsed:
            shape.append((int(source.h), int(source.w), 3))
            if source.url.startswith("rtsp"):
                procs.append(sp.Popen(['ffmpeg',
                                       '-loglevel', 'error',
                                       '-hwaccel', 'cuvid',
                                       '-c:v', 'h264_cuvid',
                                       '-rtsp_transport', 'tcp',
                                       '-i', os.path.expanduser(source.url),
                                       '-vf', 'hwdownload,format=nv12',
                                       '-c:v', 'rawvideo',
                                       '-f', 'rawvideo',
                                       '-pix_fmt', 'bgr24',
                                       'pipe:1'], stdin=sp.PIPE, stdout=sp.PIPE, shell=False, bufsize=int(source.h) * int(source.w) * 3 * 10))
            else:
                procs.append(sp.Popen(['ffmpeg',
                                       '-loglevel', 'error',
                                       '-hwaccel', 'cuvid',
                                       '-c:v', 'h264_cuvid',
                                       '-re',
                                       '-i', os.path.expanduser(source.url),
                                       '-vf', 'hwdownload,format=nv12',
                                       '-c:v', 'rawvideo',
                                       '-f', 'rawvideo',
                                       '-pix_fmt', 'bgr24',
                                       'pipe:1'], stdin=sp.PIPE, stdout=sp.PIPE, shell=False, bufsize=int(source.h) * int(source.w) * 3 * 10))
            queues.append(queue.LifoQueue(max_queue_size))
        self.shape = shape
        self.procs = procs
        self.queues = queues
        submit(self.source_queue, self.procs, self.queues)

    def source_queue(self, procs, queues):
        while True:
            for index, proc in enumerate(procs):
                try:
                    queue = queues[index]
                    h, w, c = self.shape[index]
                    frames_size = w * h * c
                    nv12_data = proc.stdout.read(frames_size)
                    if len(nv12_data) != frames_size:
                        # logger.error("source_queue read error index %s len %s", index, len(nv12_data))
                        continue
                    frame = np.frombuffer(nv12_data, dtype=np.uint8)
                    img = np.array(frame, dtype=np.uint8).reshape((h, w, c))
                    if DEBUG:
                        queue.put(img)
                    else:
                        queue.put_nowait(img)
                except Exception as e:
                    # logger.error("source_queue queue full index %s", index)
                    pass

    def produce(self):
        frames = []
        for index, q in enumerate(self.queues):
            try:
                frames.append(q.get())
            except Exception as e:
                frames.append([])
                continue
        return frames
