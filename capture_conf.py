from easydict import EasyDict

env = {
    'dev': EasyDict({
        # have little influence
        'track_max_cosine_distance': 0.6,
        'track_nn_budget': 100,
        'nms_max_overlap': 1.0,
        # important param for deep sort
        'track_max_iou_distance': 0.7,
        'track_n_init': 3,
        'track_max_age': 30,
        'min_width': 64,
        'min_height': 128,
        'frame_save_interval': 2,
        # params for lffd
        'score_threshold': 0.1,
        'top_k': 100,  # total boxes in an image
        'NMS_threshold': 0.5,
        # params for main.py
        'max_queue_size': 10,
        'pool_size': 20,
        'video_on': True,
        'track_on': True,
        'is_async': False,
        'save': True,
        'source_paths': ['./save/ETH-Jelmoli.mp4,device_2,640,480']  # ['rtsp://admin:admin123@10.27.40.47:554,device_2,1920,1080']
    }),
    'prd': EasyDict({
        # have little influence
        'track_max_cosine_distance': 0.6,
        'track_nn_budget': 100,
        'nms_max_overlap': 1.0,
        # important param for deep sort
        'track_max_iou_distance': 0.7,
        'track_n_init': 3,
        'track_max_age': 30,
        'min_width': 64,
        'min_height': 128,
        'frame_save_interval': 2,
        # params for lffd
        'score_threshold': 0.1,
        'top_k': 100,  # total boxes in an image
        'NMS_threshold': 0.5,
        # params for main.py
        'max_queue_size': 20,
        'pool_size': 40,
        'video_on': False,
        'track_on': False,
        'is_async': True,
        'save': True,
        'source_paths': ['rtsp://admin:admin123@10.27.40.47:554,device_2,1920,1080']
    })
}
from logging.config import dictConfig

dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'verbose': {
            'format': '[%(thread)d][%(levelname)s %(asctime)s %(filename)s %(funcName)s %(lineno)s] %(message)s'
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'level': 'INFO',
            'class': 'cloghandler.ConcurrentRotatingFileHandler',
            'maxBytes': 1024 * 1024 * 10,
            'backupCount': 50,
            # If delay is true,
            # then file opening is deferred until the first call to emit().
            'delay': True,
            'filename': 'logs/logs.log',
            'formatter': 'verbose',
        }
    },
    'loggers': {
        'app': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'DEBUG',
    }
}
)
