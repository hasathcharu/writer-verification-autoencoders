from modules.utils.train_utils import (
    build_global_autoencoder,
    train_writer_autoencoder,
    build_multi_conv_autoencoder,
    build_variational_autoencoder,
)
from modules.utils.eval_utils import (
    compute_autoencoder_thresholds,
    compute_conv_autoencoder_thresholds,
    evaluate_autoencoders,
    print_results,
    save_results
)

from modules.utils.common import load_parquet
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from modules.utils import constants as c
from modules.preprocess.augmentation import augment_data
import multiprocessing
import numpy as np

print("Using Features:", c.FEATURES)

if __name__ == "__main__":
    train = train_writer_autoencoder
    compute_thresh = compute_autoencoder_thresholds

    data = load_parquet(f"{c.FEATURES}/{c.PRETRAINED}/data")
    # data = load_parquet(f"data/pretrain/all")
    # print("IS NAN")
    # print(data.isna().sum())
    data = augment_data(data)
    data = augment_data(data)
    data = data.drop(columns=["sample"])
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    autoencoder = build_global_autoencoder(data.shape[1], denoising=True, lr=1e-3)
    # autoencoder, _, _ = build_variational_autoencoder(data.shape[1], denoising=True)
    # autoencoder.fit(
    #     data,
    #     data,
    #     epochs=10,
    #     batch_size=64,
    #     shuffle=True,
    #     verbose=1,
    # )

    os.makedirs(f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.PRETRAINED}", exist_ok=True)
    autoencoder.save_weights(
        f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.PRETRAINED}/pretrained.weights.h5"
    )
    joblib.dump(
        scaler, f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.PRETRAINED}/pretrained_scaler.pkl"
    )

    N_CORES = multiprocessing.cpu_count()

    dirs = sorted(os.listdir(f"{c.FEATURES}/{c.TRAIN}"))
    writer_ids = [dir.split(".")[0] for dir in dirs]
    print("Training autoencoders for writers:", len(writer_ids))
    with multiprocessing.Pool(N_CORES) as pool:
        pool.map(train, writer_ids)

    compute_thresh(writer_ids)

    global_results, sample_model = evaluate_autoencoders(writer_ids)

    print("Results:")
    print_results(global_results)
    save_results(global_results, f"{c.RESULTS}/autoencoders", sample_model, c.FEATURES)
