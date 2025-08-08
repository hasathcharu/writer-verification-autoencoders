import os
import pandas as pd
from modules.preprocess.augmentation import augment_data
from modules.utils import constants as c

os.makedirs(f"{c.FEATURES}/{c.LOCAL_TRAIN}/", exist_ok=True)

and_files = [
    os.path.join(f"{c.RAW_FEATURES}/{c.AND}", file)
    for file in sorted(os.listdir(f"{c.RAW_FEATURES}/{c.AND}"))
    if file.endswith(".csv")
]
the_files = [
    os.path.join(f"{c.RAW_FEATURES}/{c.THE}", file)
    for file in sorted(os.listdir(f"{c.RAW_FEATURES}/{c.THE}"))
    if file.endswith(".csv")
]

all_files = and_files

# writers = [dir.split(".")[0] for dir in dirs]
pre_train = pd.DataFrame()

for file in all_files:
    df = pd.read_csv(file)
    if df.empty:
        continue
    # df = augment_data(df)
    pre_train = pd.concat([pre_train, df])

pre_train.to_parquet(f"{c.FEATURES}/{c.LOCAL_TRAIN}/data.parquet", index=False)
