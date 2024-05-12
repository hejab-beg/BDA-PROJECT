import librosa
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pymongo import MongoClient
import numpy as np
import os
import concurrent.futures
from tqdm import tqdm
from joblib import Memory, Parallel, delayed
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s" 
)

DATABASE_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "music_database"
COLLECTION_NAME = "audio_features_small"

# Database setup
client = MongoClient(DATABASE_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# Generate folder names from 000 to as many as in dataset (156 for now)
# folder_names = [f"{i:03d}" for i in range(156)] 
# audio_files = []
# dataset_path = 'C:/Users/Hijab/Desktop/BDA_PROJECT/dataset'
# 
# for folder in folder_names:
#     folder_path = os.path.join(dataset_path, folder)
#     if os.path.exists(folder_path):
#         audio_files.extend(
#             [
#                 os.path.join(folder_path, file)
#                 for file in os.listdir(folder_path)
#                 if file.endswith(".mp3")
#             ]
#         )
        
base_path = '/mnt/c/Users/Hijab/Desktop/BDA_PROJECT/small_dataset'
audio_files = []
for i in range(155):
    folder_path = os.path.join(base_path, str(i).zfill(3))
    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path):
            audio_files.extend(
                [
                    os.path.join(root, file)
                    for file in files
                    if file.endswith(".mp3")
                ]
            )

# Create a memory object for caching
memory = Memory("cache_directory", verbose=0)

@memory.cache # Cache the results of the function
def process_file(file):
    try:
        y, sr = librosa.load(file, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        return mfcc
    except Exception as e:
        logging.error(f"Error loading file {file}: {e}")
        return np.array([])


# Use joblib to process the files in parallel
mfcc_features_list = Parallel(n_jobs=4)(
    delayed(process_file)(file) for file in tqdm(audio_files, total=len(audio_files))
)

# Concatenate all the MFCC features into a single 2D array
mfcc_features = np.concatenate([f.T for f in mfcc_features_list if f.size > 0], axis=0)

# Standardize the features
scaler = StandardScaler()
mfcc_scaled = scaler.fit_transform(mfcc_features)

# Fit PCA on the standardized features without reducing dimensionality.
pca = PCA()
pca.fit(mfcc_scaled)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
optimal_components = np.where(cumulative_variance >= 0.95)[0][0] + 1 

# Logging optimal components
logging.info(f"Optimal number of PCA components: {optimal_components}") 


# Processing each file in ThreadPoolExecutor
def insert_features(file, n_components):
    try:
        y, sr = librosa.load(file, sr=None)  # load the audio file and returns the audio time series (y) and sampling rate (sr).
        mfcc = librosa.feature.mfcc(y=y, sr=sr) # MFCCs are commonly used features in speech and audio processing.
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)  # used in digital signal processing to characterise a spectrum.
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y) # the rate at which the signal changes from positive to negative or back.

        scaler_mfcc = StandardScaler() # Standardize the MFCC features
        mfcc_scaled = scaler_mfcc.fit_transform(mfcc.T).T

        pca = PCA(n_components=n_components) # is a dimensionality reduction technique
        mfcc_reduced = pca.fit_transform(mfcc_scaled) # pca applied to reduce the dimensionality of the MFCC features

        scaler_feature = MinMaxScaler() # Normalize the spectral centroid and zero crossing rate
        spectral_centroid_normalized = scaler_feature.fit_transform(
            spectral_centroid.T
        ).T # normalizes the spectral centroid features by scaling them to the range [0, 1]
        zero_crossing_rate_normalized = scaler_feature.fit_transform(
            zero_crossing_rate.T
        ).T # normalizes the zero-crossing rate features by scaling them to the range [0, 1]

        document = {
            "file_name": file,
            "mfcc": mfcc_reduced.tolist(),
            "spectral_centroid": spectral_centroid_normalized.tolist(),
            "zero_crossing_rate": zero_crossing_rate_normalized.tolist(),
        }
        collection.insert_one(document)
        logging.info(f"Processed and inserted: {file}")
    except Exception as e:
        logging.error(f"Error processing file {file}: {e}")


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    for file in audio_files:
        executor.submit(insert_features, file, optimal_components)

client.close()
logging.info("MongoDB connection closed.")