import os
import cv2
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
    get_interior_contours,
    get_exterior_curves,
)
import multiprocessing

local_feature_headers = [
    "sample",
    "num_black_pixels",
    "num_interior_contours",
    "num_exterior_curves",
    "chaincode_hist_0",
    "chaincode_hist_1",
    "chaincode_hist_2",
    "chaincode_hist_3",
    "chaincode_hist_4",
    "chaincode_hist_5",
    "chaincode_hist_6",
    "chaincode_hist_7",
    # "hod_0",
    # "hod_1",
    # "hod_2",
    # "hod_3",
    # "hod_4",
    # "hod_5",
    # "hod_6",
    # "hod_7",
    # "hod_8",
    "stroke_width_hist_0",
    "stroke_width_hist_1",
    "stroke_width_hist_2",
    "stroke_width_hist_3",
    "stroke_width_hist_4",
]

WORD_DATA_PATH = f"{c.DATA_PATH}/words"
N_CORES = multiprocessing.cpu_count()


os.makedirs(f"{c.RAW_FEATURES}/{c.AND}", exist_ok=True)
os.makedirs(f"{c.RAW_FEATURES}/{c.THE}", exist_ok=True)


def get_local_features(binary_word):
    num_black_pixels = get_num_of_black_pixels(binary_word)
    binary_word_inv = cv2.bitwise_not(binary_word)
    _, num_interior_contours = get_interior_contours(binary_word_inv)
    _, num_exterior_curves = get_exterior_curves(binary_word_inv)
    chaincode_histogram, chaincode_images, color_chaincode_image = (
        compute_chaincode_histogram(binary_word_inv)
    )
    stroke_width_hist, stroke_widths = compute_stroke_width_histogram(binary_word_inv)
    hod = compute_hod(binary_word)
    data = [
        num_black_pixels,
        num_interior_contours,
        num_exterior_curves,
        *chaincode_histogram,
        # *hod,
        *stroke_width_hist,
    ]
    return (
        data,
        chaincode_images,
        color_chaincode_image,
    )


lock = threading.Lock()


def process_word(writer_folder, word, img_file):
    image = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    sample_id = img_file.split("/")[-1].split("_")[0]
    speed = img_file.split("/")[-1].split("_")[1]
    if image is None:
        print(f"Skipping {img_file} (could not load)")
        return

    (data, chaincode_images, color_chaincode_image) = get_local_features(image)

    data = pd.DataFrame([[f"{writer_folder}_{sample_id}_{speed}", *data]])

    with lock:
        data.to_csv(
            f"{c.RAW_FEATURES}/{word}/{writer_folder}.csv",
            mode="a",
            index=False,
            header=False,
        )


def process_writer_local(writer_folder, word):
    writer_path = os.path.join(WORD_DATA_PATH, writer_folder)
    if not os.path.isdir(writer_path):
        return

    images = []
    word_dir = os.path.join(writer_path, word)
    sample_dirs = [
        sample
        for sample in sorted(os.listdir(word_dir))
        if os.path.isdir(os.path.join(word_dir, sample))
    ]
    for sample in sample_dirs:
        images.extend(
            [
                os.path.join(word_dir, sample, fname)
                for fname in sorted(os.listdir(os.path.join(word_dir, sample)))
                if fname.endswith(".png")
            ]
        )

    with ThreadPoolExecutor(max_workers=N_CORES) as img_executor:
        futures = [
            img_executor.submit(process_word, writer_folder, word, img_file)
            for img_file in images
        ]

        for future in as_completed(futures):
            future.result()


writer_folders = [
    writer
    for writer in sorted(os.listdir(WORD_DATA_PATH))
    if os.path.isdir(os.path.join(WORD_DATA_PATH, writer))
]

for writer_folder in tqdm(writer_folders, desc="Extracting 'THE' Features"):
    df = pd.DataFrame(columns=local_feature_headers)
    df.to_csv(
        f"{c.RAW_FEATURES}/{c.THE}/{writer_folder}.csv",
        mode="w",
        index=False,
    )
    process_writer_local(writer_folder, c.THE)

for writer_folder in tqdm(writer_folders, desc="Extracting 'AND' Features"):
    df = pd.DataFrame(columns=local_feature_headers)
    df.to_csv(
        f"{c.RAW_FEATURES}/{c.AND}/{writer_folder}.csv",
        mode="w",
        index=False,
    )
    process_writer_local(writer_folder, c.AND)
