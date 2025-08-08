from modules.utils.train_utils import build_word_autoencoder
from modules.utils.common import load_parquet
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from modules.utils import constants as c
from tensorflow import keras
from sklearn.decomposition import PCA

# data = load_parquet(f"{c.FEATURES}/{c.LOCAL_TRAIN}/data")
data = load_parquet(f"data/and/data2")

data = data.drop(columns=["sample"])
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

autoencoder, encoder = build_word_autoencoder(
    data.shape[1], denoising=True, lr=1e-3, latent_dim=6
)

# pca = PCA(n_components=6)
# data = pca.fit_transform(data)

os.makedirs(f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.LOCAL_TRAIN}", exist_ok=True)
encoder.save(f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.LOCAL_TRAIN}/encoder.keras")
joblib.dump(scaler, f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.LOCAL_TRAIN}/scaler.pkl")
# joblib.dump(pca, f"{c.MODEL_FOLDER}/{c.AUTOENCODERS}/{c.LOCAL_TRAIN}/pca.joblib")
