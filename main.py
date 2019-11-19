# encoding: utf-8
import collections
import colorsys
import logging
import queue
import signal
import sys
from argparse import ArgumentParser
from datetime import datetime

import cv2
import numpy as np

import capture_conf
from deep_sort import nn_matching
from deep_sort import preprocessing
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from lffd.lffd import Predict
from util.cost import Cost
from util.source_queue import SourceQueue
from util.threadpoolutils import submit, new_pools

logger = logging.getLogger("app")


def sigint_handler(signum, frame):
    global is_sigint_up
    if is_sigint_up:
        sys.exit(0)
    is_sigint_up = True
    print('catched interrupt signal!')


def create_unique_color_float(tag, hue_step=0.41):
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)


Source = collections.namedtuple('Source', 'url device_id w h')


class App(object):

    def __init__(self, mode):
        self.mode = mode
        self.conf = capture_conf.env[mode]
        self.init_sources(self.conf.source_paths)
        self.detector = Predict.instance()
        self.trackers = [Tracker(nn_matching.NearestNeighborDistanceMetric("cosine", self.conf.track_max_cosine_distance, self.conf.track_nn_budget),
                                 max_iou_distance=self.conf.track_max_iou_distance,
                                 max_age=self.conf.track_max_age,
                                 n_init=self.conf.track_n_init)
                         for _ in self.sources_parsed]
        self.track_pool = new_pools(self.conf.pool_size)
        self.save_pool = new_pools(self.conf.pool_size)
        self.frame_index = 0
        self.video_state = False
        if self.conf.video_on:
            self.box_queue = queue.LifoQueue(100)
            if self.conf.is_async:
                submit(self.video_on)
        self.debug = mode == 'dev'
        if self.debug:
            self.last_time = datetime.now()
            self.fps = 0
            self.pids = set()

    def init_sources(self, source_path):
        logger.info("init_sources %s", source_path)
        sources_parsed = [Source(*s.split(",")) for s in source_path]
        self.sources_parsed = sources_parsed
        self.sourceQueue = SourceQueue(sources_parsed, max_queue_size=self.conf.max_queue_size)

    def test(self):
        for i in range(1, 9):
            im = cv2.imread('/home/lijc08/桌面/{}.jpg'.format(i))
            now = datetime.now()
            bboxes, feature = self.detector.predict(im, score_threshold=self.conf.score_threshold, top_k=self.conf.top_k, NMS_threshold=self.conf.NMS_threshold)
            print('cost:{} sec'.format((datetime.now() - now).total_seconds()))
            for bbox in bboxes:
                cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            if max(im.shape[:2]) > 1440:
                scale = 1440 / max(im.shape[:2])
                im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
            cv2.imshow('im', im)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()

    def video_on(self):
        if self.conf.is_async:
            if not self.video_state:
                self.video_state = True
                while True:
                    self.video_on_sync()
        else:
            self.video_on_sync()

    def video_on_sync(self):
        cost = Cost("video display")
        im, bboxes, tracks = self.box_queue.get()
        for bbox in bboxes:
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
        cost.record("draw detect")
        if self.conf.track_on:
            for track in tracks:
                color = create_unique_color_uchar(track[1])
                x1, y1, x2, y2 = self.tlwh2rec(track[2])
                cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
        cost.record("draw track")
        if max(im.shape[:2]) > 1440:
            scale = 1440 / max(im.shape[:2])
            im = cv2.resize(im, (0, 0), fx=scale, fy=scale)
        cv2.putText(im, str(format(self.fps, '.2f')) + " fps", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2, lineType=2)
        cv2.imshow('detect', im)
        cv2.waitKey(1)
        cost.end(func=logger.info, show=True)

    def detectVideo(self):
        cost = Cost("detectVideo")
        imgs = self.sourceQueue.produce()
        cost.record("produceFrames")
        frame_count = 0
        for i, im in enumerate(imgs):
            if len(im) == 0:
                continue
            frame_count += 1
            try:
                bboxes, feature = self.detector.predict(im, score_threshold=self.conf.score_threshold, top_k=self.conf.top_k, NMS_threshold=self.conf.NMS_threshold)
                if bboxes is None or len(bboxes) == 0:
                    continue
                if self.conf.is_async:
                    self.track_pool.submit(self.track, self.trackers[i], im, bboxes, feature, self.frame_index)
                else:
                    self.track(self.trackers[i], im, bboxes, feature, self.frame_index)
            except Exception as e:
                logger.error(e, exc_info=True)
        self.frame_index = 0 if self.frame_index > sys.maxsize else self.frame_index + 1
        cost.record("detect")
        if self.debug:
            now = datetime.now()
            self.fps = frame_count / (now - self.last_time).total_seconds()
            self.last_time = now
            cost.end(logger.info, show=True)

    def track(self, tracker, im, bboxes, feature, frame_index):
        try:
            cost = Cost("track")
            detections = self.create_detections(bboxes, feature)
            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, self.conf.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Update tracker.
            tracker.predict()
            tracker.update(detections)

            # Store confirmed track into results.
            results = []
            if self.frame_index % self.conf.frame_save_interval == 0:
                tracker.last_track_ids.clear()
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                if track.track_id not in tracker.last_track_ids:
                    results.append([frame_index, track.track_id, track.to_tlwh()])
                    tracker.last_track_ids.add(track.track_id)
            if self.conf.video_on:
                self.box_queue.put((im.copy(), bboxes, results))
                self.video_on()
            if self.debug:
                [self.pids.add(r[1]) for r in results]
                logger.info("pid counts:%s", len(self.pids))
                cost.end(logger.info, show=True)
            if self.conf.save:
                self.save_images(img=im, tracks=results)
        except Exception as e:
            logger.error(e, exc_info=True)

    def create_detections(self, bboxes, feature):
        detections = []
        for box in np.array(bboxes):
            if box is None or len(box) == 0:
                continue
            box[2:4] -= box[:2]
            # too small to do reid
            if box[2] < self.conf.min_width and box[3] < self.conf.min_height:
                continue
            detections.append(Detection(tlwh=box[:4], confidence=box[4], feature=[]))
        return detections

    def tlwh2rec(self, tlwh):
        x1, y1 = max(int(tlwh[0]), 0), max(int(tlwh[1]), 0)
        x2, y2 = int(x1 + tlwh[2]), int(y1 + tlwh[3])
        return x1, y1, x2, y2

    def save_images(self, img, tracks):
        # todo save remote
        cost = Cost("save_images")
        for track in tracks:
            frame_index, track_id = track[0], track[1]
            x1, y1, x2, y2 = self.tlwh2rec(track[2])
            ret_img = img[y1:y2, x1:x2, :]
            if self.conf.is_async:
                self.save_pool.submit(cv2.imwrite, './save/images/{}_{}.jpg'.format(track_id, frame_index), ret_img)
            else:
                if self.debug:
                    cv2.imshow('save', ret_img)
                cv2.imwrite('./save/images/{}_{}.jpg'.format(track_id, frame_index), ret_img)
        cost.end(logger.info, show=True if self.debug else False)

    def clear(self):
        cv2.destroyAllWindows()


def parseArgvs():
    parser = ArgumentParser(description='capture service')
    parser.add_argument("--mode", type=str, help="mode", choices=["dev", "prd"], default="dev")
    args = parser.parse_args()
    logger.info(args)
    return args


if __name__ == '__main__':
    import os
    from mxnet import context

    logger.info("num gpu:%s", context.num_gpus())
    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    args = parseArgvs()
    logger.info('mode %s ', args.mode)
    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGHUP, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)
    is_sigint_up = False
    app = App(args.mode)
    # given video path, predict and show
    while not is_sigint_up:
        app.detectVideo()
        # app.test()
    logger.info("while end")
    app.clear()
