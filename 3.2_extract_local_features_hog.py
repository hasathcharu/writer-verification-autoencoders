import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from modules.utils import constants as c
from modules.feature_extraction.features import (
    compute_hod,
    compute_stroke_width_histogram,
    get_num_of_black_pixels,
    compute_chaincode_histogram,
    get_contour_areas,
    compute_hog
)
import multiprocessing

local_feature_headers = [
    "sample",
    *[f"hog_{i}" for i in range(9)],
    # "interior_contour_area",
]

CHAR_DATA_PATH = f"{c.DATA_PATH}/{c.CHARS}"
N_CORES = multiprocessing.cpu_count()

os.makedirs(f"{c.RAW_FEATURES}/{c.CHARS}", exist_ok=True)


def get_local_features(binary_char):
    hog_hist = compute_hog(binary_char)
    binary_inv = cv2.bitwise_not(binary_char)
    # _, interior_contour_area = get_contour_areas(binary_inv)
    data = [
        *hog_hist,
        # interior_contour_area,
    ]
    return data


lock = threading.Lock()


def process_char(writer_folder, img_file):
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    sample_id = img_file.split("/")[-1].split("_")[1]
    speed = img_file.split("/")[-1].split("_")[2]
    line_id = img_file.split("/")[-1].split("_")[3]

    if image is None:
        print(f"Skipping {img_file} (could not load)")
        return

    data = get_local_features(image)

    data = pd.DataFrame([[f"{writer_folder}_{sample_id}_{speed}_{line_id}", *data]])

    with lock:
        data.to_csv(
            f"{c.RAW_FEATURES}/{c.CHARS}/{writer_folder}.csv",
            mode="a",
            index=False,
            header=False,
        )


def process_writer_local(writer_folder):
    writer_path = os.path.join(CHAR_DATA_PATH, writer_folder)
    if not os.path.isdir(writer_path):
        return

    images = []
    sample_dirs = [
        sample
        for sample in sorted(os.listdir(writer_path))
        if os.path.isdir(os.path.join(writer_path, sample))
    ]
    for sample in sample_dirs:
        sample_path = os.path.join(writer_path, sample)
        speed_dirs = [
            speed
            for speed in sorted(os.listdir(sample_path))
            if os.path.isdir(os.path.join(sample_path, speed))
        ]
        for speed in speed_dirs:
            speed_path = os.path.join(sample_path, speed)
            line_dirs = [
                line
                for line in sorted(os.listdir(speed_path))
                if os.path.isdir(os.path.join(speed_path, line))
            ]
            for line in line_dirs:
                line_path = os.path.join(speed_path, line)
                images.extend(
                    [
                        os.path.join(line_path, fname).replace('\\', '/')
                        for fname in sorted(os.listdir(line_path))
                        if fname.endswith(".png")
                    ]
                )

    with ThreadPoolExecutor(max_workers=N_CORES) as img_executor:
        futures = [
            img_executor.submit(process_char, writer_folder, img_file)
            for img_file in images
        ]

        for future in as_completed(futures):
            future.result()


writer_folders = [
    writer
    for writer in sorted(os.listdir(CHAR_DATA_PATH))
    if os.path.isdir(os.path.join(CHAR_DATA_PATH, writer))
]

for writer_folder in tqdm(writer_folders, desc="Extracting Character Features"):
    df = pd.DataFrame(columns=local_feature_headers)
    df.to_csv(
        f"{c.RAW_FEATURES}/{c.CHARS}/{writer_folder}.csv",
        mode="w",
        index=False,
    )
    process_writer_local(writer_folder)

