import os
import pandas as pd
from modules.utils import constants as c
from modules.utils.common import load_model, load_scaler
from tqdm import tqdm

os.makedirs(f"{c.FEATURES}/{c.LOCAL_PRED}/{c.AND}", exist_ok=True)
os.makedirs(f"{c.FEATURES}/{c.LOCAL_PRED}/{c.THE}", exist_ok=True)


and_columns = [
    "and_latent_0",
    "and_latent_1",
    "and_latent_2",
    "and_latent_3",
    "and_latent_4",
    "and_latent_5",
]
the_columns = [
    "the_latent_0",
    "the_latent_1",
    "the_latent_2",
    "the_latent_3",
    "the_latent_4",
    "the_latent_5",
]

dirs = sorted(os.listdir(f"{c.RAW_FEATURES}/{c.AND}"))
writers = [dir.split(".")[0] for dir in dirs]

# word_encoder = load_model(f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.LOCAL_TRAIN}/encoder")
word_encoder = load_model(f"data/cvl/models/{c.AUTOENCODERS}/{c.LOCAL_TRAIN}/encoder")
# scaler = load_scaler(f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.LOCAL_TRAIN}/scaler")
scaler = load_scaler(f"data/cvl/models/{c.AUTOENCODERS}/{c.LOCAL_TRAIN}/scaler")

for writer in tqdm(writers, desc="Preparing Local Predictions"):

    and_df = pd.read_csv(f"{c.RAW_FEATURES}/{c.AND}/{writer}.csv")
    # the_df = pd.read_csv(f"{c.RAW_FEATURES}/{c.THE}/{writer}.csv")

    pd.DataFrame(columns=["sample", *and_columns]).to_parquet(
        f"{c.FEATURES}/{c.LOCAL_PRED}/{c.AND}/{writer}.parquet", index=False
    )
    # pd.DataFrame(columns=["sample", *the_columns]).to_parquet(
    #     f"{c.FEATURES}/{c.LOCAL_PRED}/{c.THE}/{writer}.parquet", index=False
    # )

    if and_df.shape[0] > 0:
        samples = and_df["sample"]
        and_data = and_df.drop(columns=["sample"])
        and_data = scaler.transform(and_data)
        encoded = word_encoder.predict(and_data, verbose=0)
        encoded = pd.DataFrame(encoded, columns=and_columns)
        encoded = pd.concat([samples, encoded], axis=1)
        encoded.to_parquet(
            f"{c.FEATURES}/{c.LOCAL_PRED}/{c.AND}/{writer}.parquet", index=False
        )

    # if the_df.shape[0] > 0:
    #     samples = the_df["sample"]
    #     the_data = the_df.drop(columns=["sample"])
    #     the_data = scaler.transform(the_data)
    #     encoded = word_encoder.predict(the_data, verbose=0)
    #     encoded = pd.DataFrame(encoded, columns=the_columns)
    #     encoded = pd.concat([samples, encoded], axis=1)
    #     encoded.to_parquet(
    #         f"{c.FEATURES}/{c.LOCAL_PRED}/{c.THE}/{writer}.parquet", index=False
    #     )
