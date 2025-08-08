from modules.utils import constants as c
import os
import random

LINE_DATA_PATH = f"{c.DATA_PATH}/binary_lines"

writers = [
    writer
    for writer in sorted(os.listdir(LINE_DATA_PATH))
    if os.path.isdir(os.path.join(LINE_DATA_PATH, writer))
]

pretrain_writers = random.sample(writers, int(len(writers) * 0.3))
writers = [w for w in writers if w not in pretrain_writers]

os.makedirs(f"{c.FEATURES}/", exist_ok=True)

with open(f"{c.FEATURES}/pretrain.txt", "w") as f:
    for writer in pretrain_writers:
        f.write(f"{writer}\n")

with open(f"{c.FEATURES}/train.txt", "w") as f:
    for writer in writers:
        f.write(f"{writer}\n")
