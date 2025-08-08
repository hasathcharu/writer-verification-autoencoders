import os
import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from modules.utils import constants as c
from skimage.measure import shannon_entropy
from modules.feature_extraction.features import (
    compute_stroke_width_histogram,
    get_num_of_black_pixels,
    get_interior_contours,
    get_exterior_curves,
    compute_chaincode_histogram,
    get_global_word_features,
    get_zone_features,
    contour_direction_pdf,
    compute_slant_angle_histogram,
    contour_hinge_pdf
)

global_feature_headers = [
    "sample",
    "num_black_pixels",
    "grey_level_threshold",
    "grey_entropy",
    "num_interior_contours",
    "mean_interior_area",
    "num_exterior_curves",
    *[f"stroke_width_hist_{i}" for i in range(6)],
    "mean_word_gap",
    "std_word_gap",
    "num_words",
    *[f"chaincode_hist_{i}" for i in range(8)],
    *[f"contour_hinge_pca_{i}" for i in range(15)],
    *[f"slant_angle_hist_{i}" for i in range(9)],
    "viz_upper",
    "viz_middle",
    "viz_lower",
]

LINE_DATA_PATH = f"{c.DATA_PATH}/raw_lines"
DIRTY_LINE_DATA_PATH = f"{c.DATA_PATH}/dirty_binary_lines"
os.makedirs(f"{c.RAW_FEATURES}/{c.GLOBAL}", exist_ok=True)

def get_global_features(sample_name, raw_line, dirty_binary_line, contour_hinge_pca):
    grey_level_threshold, binary_line = cv2.threshold(
        raw_line, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    grey_entropy = shannon_entropy(raw_line)
    num_black_pixels = get_num_of_black_pixels(binary_line)
    line_overlap = max(0, get_num_of_black_pixels(dirty_binary_line) - num_black_pixels)
    binary_line_inv = cv2.bitwise_not(binary_line)
    interior_contours, num_interior_contours, mean_interior_area, std_interior_area = get_interior_contours(binary_line_inv)
    exterior_curves, num_exterior_curves = get_exterior_curves(binary_line_inv)
    chaincode_histogram, chaincode_images, color_chaincode_image = (
        compute_chaincode_histogram(binary_line_inv)
    )
    # contour_direction_hist = contour_direction_pdf(binary_line_inv)
    stroke_width_hist, stroke_widths = compute_stroke_width_histogram(binary_line_inv)
    gap_line, gaps, num_words = get_global_word_features(binary_line)
    viz_upper, viz_middle, viz_lower = get_zone_features(binary_line)
    # run_length_pca_data = run_length_pca[run_length_pca["sample"] == sample_name].iloc[0, 1:].values
    contour_hinge_pca_data = contour_hinge_pca[contour_hinge_pca["sample"] == sample_name].iloc[0, 1:].values
    slant_angle_hist = compute_slant_angle_histogram(binary_line_inv)

    data = [
        num_black_pixels,
        grey_level_threshold,
        grey_entropy,
        num_interior_contours,
        mean_interior_area,
        num_exterior_curves,
        *stroke_width_hist,
        np.mean(gaps) if len(gaps) > 0 else 0,
        np.std(gaps) if len(gaps) > 0 else 0,
        num_words,
        # line_overlap,
        *chaincode_histogram,
        *contour_hinge_pca_data,
        *slant_angle_hist,
        viz_upper,
        viz_middle,
        viz_lower,
    ]
    return (
        data,
        interior_contours,
        exterior_curves,
        chaincode_images,
        color_chaincode_image,
        gap_line,
        stroke_width_hist,
    )


lock = threading.Lock()


def process_line(writer_folder, line_path, contour_hinge_pca):
    line = cv2.imread(os.path.join(LINE_DATA_PATH, line_path), cv2.IMREAD_GRAYSCALE)
    dirty_binary_line = cv2.imread(
        os.path.join(DIRTY_LINE_DATA_PATH, line_path), cv2.IMREAD_GRAYSCALE
    )
    sample_id = line_path.split("/")[1]
    speed = line_path.split("/")[2]
    line_num = line_path.split("/")[3].split(".")[0]
    sample_name = f"{writer_folder}_{sample_id}_{speed}_{line_num}"
    if line is None or dirty_binary_line is None:
        print(f"Skipping {line_path} (could not load)")
        return

    (
        data,
        interior_contours,
        exterior_curves,
        chaincode_images,
        color_chaincode_image,
        gap_line,
        stroke_width_hist,
    ) = get_global_features(sample_name, line, dirty_binary_line, contour_hinge_pca)

    data = pd.DataFrame([[sample_name, *data]])
    with lock:
        data.to_csv(
            f"{c.RAW_FEATURES}/{c.GLOBAL}/{writer_folder}.csv",
            mode="a",
            header=False,
            index=False,
        )
        # visualize(raw_line, binary_line, interior_contours, exterior_curves, slope_components, gap_line)


def process_writer_global(writer_folder):
    writer_path = os.path.join(LINE_DATA_PATH, writer_folder)
    contour_hinge_pca = pd.read_csv(
        f"{c.RAW_FEATURES}/{c.PCA}/contour_hinge/{writer_folder}.csv"
    )

    images = [
        os.path.join(writer_folder, sample, speed, img_file).replace("\\", "/")
        for sample in sorted(os.listdir(writer_path))
        if os.path.isdir(os.path.join(writer_path, sample))
        for speed in os.listdir(os.path.join(writer_path, sample))
        if os.path.isdir(os.path.join(writer_path, sample, speed))
        for img_file in os.listdir(os.path.join(writer_path, sample, speed))
        if img_file.endswith(".png")
    ]

    with ThreadPoolExecutor(max_workers=8) as img_executor:
        futures = [
            img_executor.submit(process_line, writer_folder, img_file, contour_hinge_pca)
            for img_file in images
        ]

        for future in as_completed(futures):
            future.result()


writer_folders = [
    writer
    for writer in sorted(os.listdir(LINE_DATA_PATH))
    if os.path.isdir(os.path.join(LINE_DATA_PATH, writer))
]


for writer_folder in tqdm(writer_folders, desc="Extracting Global Features"):
    df = pd.DataFrame(columns=global_feature_headers)
    df.to_csv(
        f"{c.RAW_FEATURES}/{c.GLOBAL}/{writer_folder}.csv",
        mode="w",
        index=False,
    )
    process_writer_global(writer_folder)
