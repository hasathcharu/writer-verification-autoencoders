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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from modules.feature_extraction.features import run_length_pdf

num_components = 20

global_feature_headers = [
    "sample",
    *[f"run_length_{i}" for i in range(120)],
]

pca_feature_headers = [
    "sample",
    *[f"run_length_pca_{i}" for i in range(num_components)],
]

LINE_DATA_PATH = f"{c.DATA_PATH}/raw_lines"

os.makedirs(f"{c.RAW_FEATURES}/{c.GLOBAL}", exist_ok=True)
os.makedirs(f"{c.RAW_FEATURES}/{c.HIST}", exist_ok=True)
os.makedirs(f"{c.RAW_FEATURES}/{c.PCA}/run_length", exist_ok=True)
os.makedirs(f"{c.MODEL_FOLDER}/{c.PCA}", exist_ok=True)

lock = threading.Lock()


def process_line(writer_folder, line_path):
    line = cv2.imread(os.path.join(LINE_DATA_PATH, line_path), cv2.IMREAD_GRAYSCALE)
    sample_id = line_path.split("/")[1]
    speed = line_path.split("/")[2]
    line_num = line_path.split("/")[3].split(".")[0]
    if line is None:
        print(f"Skipping {line_path} (could not load)")
        return

    _, binary_line = cv2.threshold(line, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary_line_inv = cv2.bitwise_not(binary_line)
    run_length_hist_rows = run_length_pdf(binary_line_inv, axis=0)
    run_length_hist_cols = run_length_pdf(binary_line_inv, axis=1)
    run_length_hist = np.concatenate((run_length_hist_rows, run_length_hist_cols))

    data = pd.DataFrame(
        [[f"{writer_folder}_{sample_id}_{speed}_{line_num}", *run_length_hist]]
    )
    with lock:
        data.to_csv(
            f"{c.RAW_FEATURES}/{c.HIST}/run_length_hist.csv",
            mode="a",
            header=False,
            index=False,
        )


def process_line_pca(writer_folder, line_path, pca, scaler):
    line = cv2.imread(os.path.join(LINE_DATA_PATH, line_path), cv2.IMREAD_GRAYSCALE)

    sample_id = line_path.split("/")[1]
    speed = line_path.split("/")[2]
    line_num = line_path.split("/")[3].split(".")[0]
    if line is None:
        print(f"Skipping {line_path} (could not load)")
        return

    _, binary_line = cv2.threshold(line, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary_line_inv = cv2.bitwise_not(binary_line)
    run_length_hist_rows = run_length_pdf(binary_line_inv, axis=0)
    run_length_hist_cols = run_length_pdf(binary_line_inv, axis=1)
    run_length_hist = np.concatenate((run_length_hist_rows, run_length_hist_cols))
    # run_length_hist = scaler.transform([run_length_hist])
    # run_length_hist = pca.transform(run_length_hist)
    # run_length_hist = run_length_hist.flatten()

    data = pd.DataFrame(
        [[f"{writer_folder}_{sample_id}_{speed}_{line_num}", *run_length_hist]]
    )
    with lock:
        data.to_csv(
            f"{c.RAW_FEATURES}/{c.PCA}/run_length/{writer_folder}.csv",
            mode="a",
            header=False,
            index=False,
        )


def process_writer_global(writer_folder):
    writer_path = os.path.join(LINE_DATA_PATH, writer_folder)

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
            img_executor.submit(process_line, writer_folder, img_file)
            for img_file in images
        ]

        for future in as_completed(futures):
            future.result()


def process_writer_pca_global(writer_folder, pca, scaler):
    writer_path = os.path.join(LINE_DATA_PATH, writer_folder)

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
            img_executor.submit(process_line_pca, writer_folder, img_file, pca, scaler)
            for img_file in images
        ]

        for future in as_completed(futures):
            future.result()


writers = [line.strip() for line in open(f"{c.FEATURES}/pretrain.txt", "r")]

df = pd.DataFrame(columns=global_feature_headers)
df.to_csv(
    f"{c.RAW_FEATURES}/{c.HIST}/run_length_hist.csv",
    mode="w",
    index=False,
)

for writer_folder in tqdm(writers, desc="Extracting Run Length Features"):
    process_writer_global(writer_folder)

data = pd.read_csv(f"{c.RAW_FEATURES}/{c.HIST}/run_length_hist.csv")
data = data.drop(columns=["sample"])

scaler = StandardScaler()
data = scaler.fit_transform(data.values)

pca = PCA(n_components=num_components)
pca.fit(data)

joblib.dump(pca, f"{c.MODEL_FOLDER}/{c.PCA}/run_length_hist_pca.pkl")
joblib.dump(scaler, f"{c.MODEL_FOLDER}/{c.PCA}/run_length_hist_scaler.pkl")

writers.extend([line.strip() for line in open(f'{c.FEATURES}/train.txt', 'r')])

for writer_folder in tqdm(writers, desc="Creating PCA Features"):
    df = pd.DataFrame(columns=pca_feature_headers)
    df.to_csv(
        f"{c.RAW_FEATURES}/{c.PCA}/run_length/{writer_folder}.csv",
        mode="w",
        index=False,
    )
    process_writer_pca_global(writer_folder, pca, scaler)
