import os
import torch
from tqdm import tqdm
from modules.preprocess.preprocess import read_image, remove_rules
from modules.preprocess.segmentation import clean_lines, segment_words, segment_lines
import cv2
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.utils import constants as c
import multiprocessing

N_CORES = multiprocessing.cpu_count()

DATASET_PATH = f"{c.ORIG_DATA_PATH}/cropped"

BINARY_OUTPUT_PATH = f"{c.DATA_PATH}/binary"
RAW_OUTPUT_PATH = f"{c.DATA_PATH}/raw"
WORD_OUTPUT_PATH = f"{c.DATA_PATH}/words"
BINARY_LINE_OUTPUT_PATH = f"{c.DATA_PATH}/binary_lines"
RAW_LINE_OUTPUT_PATH = f"{c.DATA_PATH}/raw_lines"
DIRTY_BINARY_LINE_OUTPUT_PATH = f"{c.DATA_PATH}/dirty_binary_lines"
DIRTY_RAW_LINE_OUTPUT_PATH = f"{c.DATA_PATH}/dirty_raw_lines"

writer_folders = sorted(os.listdir(DATASET_PATH))
total_images = sum(
    len(os.listdir(os.path.join(DATASET_PATH, wf)))
    for wf in writer_folders
    if os.path.isdir(os.path.join(DATASET_PATH, wf))
)

lock = threading.Lock()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(writer_folder, img_file):
    img_path = os.path.join(DATASET_PATH, writer_folder, img_file)
    img_name, img_ext = os.path.splitext(img_file)
    sample_id = img_name.split("_")[1]
    speed = img_name.split("_")[2]
    binary_output_folder = f"{BINARY_OUTPUT_PATH}/{writer_folder}"
    raw_output_folder = f"{RAW_OUTPUT_PATH}/{writer_folder}"
    binary_line_output_folder = (
        f"{BINARY_LINE_OUTPUT_PATH}/{writer_folder}/{sample_id}/{speed}"
    )
    raw_line_output_folder = (
        f"{RAW_LINE_OUTPUT_PATH}/{writer_folder}/{sample_id}/{speed}"
    )
    dirty_binary_line_output_folder = (
        f"{DIRTY_BINARY_LINE_OUTPUT_PATH}/{writer_folder}/{sample_id}/{speed}"
    )
    dirty_raw_line_output_folder = (
        f"{DIRTY_RAW_LINE_OUTPUT_PATH}/{writer_folder}/{sample_id}/{speed}"
    )

    image = read_image(img_path)
    raw_image, binary_image = remove_rules(image)

    (lines, dirty_lines), cropped_rectangles = segment_lines(binary_image)
    dirty_raw_lines = [
        raw_image[y1:y2, x1:x2] for (x1, y1, x2, y2) in cropped_rectangles
    ]
    raw_lines, _ = clean_lines(dirty_raw_lines)

    with lock:
        os.makedirs(binary_output_folder, exist_ok=True)
        os.makedirs(raw_output_folder, exist_ok=True)

        os.makedirs(binary_line_output_folder, exist_ok=True)
        os.makedirs(raw_line_output_folder, exist_ok=True)
        os.makedirs(dirty_binary_line_output_folder, exist_ok=True)
        os.makedirs(dirty_raw_line_output_folder, exist_ok=True)

        os.makedirs(
            f"{WORD_OUTPUT_PATH}/{writer_folder}/{c.AND}/{sample_id}", exist_ok=True
        )
        os.makedirs(
            f"{WORD_OUTPUT_PATH}/{writer_folder}/{c.THE}/{sample_id}", exist_ok=True
        )

        cv2.imwrite(os.path.join(binary_output_folder, img_file), binary_image)
        cv2.imwrite(os.path.join(raw_output_folder, img_file), raw_image)

        for i, line in enumerate(lines):
            cv2.imwrite(f"{binary_line_output_folder}/{i}.png", line)
        for i, line in enumerate(raw_lines):
            cv2.imwrite(f"{raw_line_output_folder}/{i}.png", line)
        for i, line in enumerate(dirty_lines):
            cv2.imwrite(f"{dirty_binary_line_output_folder}/{i}.png", line)
        for i, line in enumerate(dirty_raw_lines):
            cv2.imwrite(f"{dirty_raw_line_output_folder}/{i}.png", line)


with tqdm(
    total=total_images, desc="Preprocessing", position=0, leave=True
) as progress_bar:
    for writer_folder in writer_folders:
        writer_path = os.path.join(DATASET_PATH, writer_folder)
        if not os.path.isdir(writer_path) or not writer_folder.startswith("W"):
            continue
        images = sorted(os.listdir(writer_path))
        with ThreadPoolExecutor(max_workers=N_CORES) as img_executor:
            futures = [
                img_executor.submit(preprocess, writer_folder, img_file)
                for img_file in images
            ]

            for future in as_completed(futures):
                future.result()
                progress_bar.update(1)
