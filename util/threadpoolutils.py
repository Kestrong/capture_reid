from concurrent.futures import ThreadPoolExecutor
import os
from threading import Condition
import time


class CountDownLatch:

    def __init__(self, count):
        self.count = count
        self.condition = Condition()

    def await_(self):
        try:
            self.condition.acquire()
            while self.count > 0:
                self.condition.wait()
        finally:
            self.condition.release()

    def await_after(self, timeout):
        try:
            begin = time.time()
            while begin + timeout >= time.time() and self.count > 0:
                time.sleep(1)
        finally:
            pass

    def countDown(self):
        try:
            self.condition.acquire()
            self.count -= 1
            if self.count <= 0:
                self.condition.notifyAll()
        finally:
            self.condition.release()

    def getCount(self):
        return self.count


def new_pools(max_workers=(os.cpu_count() or 1) * 5):
    return ThreadPoolExecutor(max_workers=max_workers)


__pools = new_pools()


def submit(fn, *args, **kwargs):
    __pools.submit(fn, *args, **kwargs)


if __name__ == "__main__":
    pool = new_pools()
    latch = CountDownLatch(10)
    cb = lambda x: (print(x + 1), latch.countDown())
    for i in range(10):
        pool.submit(cb, i)
    latch.await_()
    print('end')
