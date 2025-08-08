import os
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from modules.utils import constants as c
import numpy as np
from tqdm import tqdm
from modules.preprocess.augmentation import augment_data

os.makedirs(f"{c.FEATURES}/sample/", exist_ok=True)

csvs = [
    csv
    for csv in sorted(os.listdir(f"{c.RAW_FEATURES}/{c.GLOBAL}"))
    if csv.endswith(".csv")
]

writers = [line.strip() for line in open(f'{c.FEATURES}/train.txt', 'r')]

def get_local_features(writer, df):
    df = df.copy()

    char_df = pd.read_csv(f"{c.RAW_FEATURES}/{c.CHARS}/{writer}.csv")

    char_out_df = pd.DataFrame(
        np.zeros((df.shape[0], len(char_df.columns[1:]))), columns=char_df.columns[1:]
    )

    for i, row in df.iterrows():
        sample_id = row["sample"]
        char_rows = char_df[char_df["sample"] == sample_id]
        if len(char_rows) > 0:
            char_data = char_rows.drop(columns=["sample"])
            char_out_df.iloc[i] = char_data.mean(axis=0)
        else:
            sample_id = "_".join(sample_id.split("_")[:-1])
            char_rows = char_df[char_df["sample"].str.startswith(sample_id)]
            char_data = char_rows.drop(columns=["sample"]).sample(n=min(4, len(char_rows)))
            char_out_df.iloc[i] = char_data.mean(axis=0)
    char_out_df = char_out_df.add_prefix("e_")
    df = pd.concat([df, char_out_df], axis=1)
    return df

for writer in tqdm(writers, desc="Preparing Global Data"):
    df = pd.read_csv(f"{c.RAW_FEATURES}/{c.GLOBAL}/{writer}.csv")

    if df.shape[0] < 3:
        print(f"[TRAIN] Skipping {writer} due to insufficient data")
        continue
    
    df = get_local_features(writer, df)
    samples = df["sample"]
    sample_ids = [sample.split("_")[1] for sample in samples]
    test_sample = random.sample(sample_ids, 1)[0]
    test_own = df[df["sample"].str.contains(test_sample)]
    train = df[~df["sample"].str.contains(test_sample)]
    
    train, val_own = train_test_split(train, test_size=0.2, random_state=42)

    # Make a copy of the list of writers
    test_writers = writers.copy()
    # Remove the current writer from the list
    test_writers.remove(writer)

    # Sample 50 other writers for validation
    val_other_writers = random.sample(test_writers, 30)
    # Take the rest of the writers for testing
    test_other_writers = [w for w in test_writers if w not in val_other_writers]

    val = val_own.copy()
    # val = augment_data(val)
    val["label"] = 0

    test = test_own.copy()
    test["label"] = 0

    val_other = pd.DataFrame()

    for _writer in val_other_writers:
        data = pd.read_csv(f"{c.RAW_FEATURES}/{c.GLOBAL}/{_writer}.csv")
        if data.shape[0] == 0:
            print(f"[VAL] Skipping {_writer} due to insufficient data")
            continue
        data = get_local_features(_writer, data)
        data["label"] = 1
        val_other = pd.concat([val_other, data])
    val = pd.concat([val, val_other.sample(n=len(val), random_state=42)])

    test_other = pd.DataFrame()
    for _writer in test_other_writers:
        data = pd.read_csv(f"{c.RAW_FEATURES}/{c.GLOBAL}/{_writer}.csv")
        if data.shape[0] == 0:
            print(f"[TEST] Skipping {_writer} due to insufficient data")
            continue
        data["label"] = 1
        data = get_local_features(_writer, data)
        test_other = pd.concat([test_other, data])

    for _ in range(2):
        random_sample = test_other.sample(n=1)
        print(random_sample["sample"])
    test = pd.concat([test, test_other.sample(n=len(test), random_state=42)])

    # test.to_parquet(f"{c.FEATURES}/sample/{writer}.parquet", index=False)
