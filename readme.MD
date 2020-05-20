### 行人检测和识别
可基于摄像头实时监控或录制的视频或静态图片检测行人并跟踪。
#### 行人检测
检测算法使用 [LFFD](https://github.com/YonghaoHe/A-Light-and-Fast-Face-Detector-for-Edge-Devices) ，跟踪算法使用[deep sort](https://github.com/nwojke/deep_sort) ，主程序入口main.py。
#### 行人重识别
使用 [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline) 算法进行识别，主程序入口reid.py。