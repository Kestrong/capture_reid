# -*- coding: utf-8 -*-
import time


class Cost:
    def __init__(self, eventName="default"):
        self.eventName = eventName
        self.events = []
        self.started = time.time()

    def reset(self):
        self.events = []
        self.started = time.time()

    def record(self, name):
        self.events.append((name, time.time()))

    def end(self, func=None, show=True, maxTotal=0):
        if not show:
            return

        costs = {}
        last = self.started
        for event in self.events:
            name, t = event
            c = t - last
            if name in costs:
                costs[name] += c
            else:
                costs[name] = c
            last = t
        total = time.time() - self.started

        logs = "[{}]=>".format(self.eventName)
        for name, cost_time in costs.items():
            logs += "[{}]:{:.2f}ms,".format(name, cost_time * 1000)
        logs += "total_cost:{:.2f}ms".format(total * 1000)

        if maxTotal > 0 and total < maxTotal:
            return
        if func:
            func(logs)
        else:
            print(logs)
        self.reset()


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor


    def test(index):
        try:
            c = Cost(str(index))
            time.sleep(1)
            c.record("1")
            time.sleep(2)
            c.record("2")
            c.end(maxTotal=1)
        except Exception as e:
            print(e)


    executor = ThreadPoolExecutor(max_workers=2)
    executor.submit(test, 1)
    executor.submit(test, 2)
    executor.submit(test, 3)

    time.sleep(5)
