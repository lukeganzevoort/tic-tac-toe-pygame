import multiprocessing as mp

from agent import train


def train_multiple():
    procs: dict[int, mp.Process] = {}
    for i in range(16):
        procs[i] = mp.Process(target=train)
    for proc in procs.values():
        proc.start()
    for proc in procs.values():
        proc.join()


if __name__ == "__main__":
    train_multiple()
