import multiprocessing as mp

from agent import train


def train_multiple():
    procs = []
    for _ in range(2):
        procs.append(mp.Process(target=train))
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()
