from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import flatten, array, col
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import DenseVector
from annoy import AnnoyIndex
import numpy as np


# create a Spark session and modify the configuration settings for larger datasets
print("Creating Spark session...")
spark = SparkSession.builder \
    .appName("Music Recommendation System") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.cores", "4") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/yourdb.yourcollection") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/yourdb.yourcollection") \
    .getOrCreate()

# load the data from MongoDB
print("Loading data from MongoDB...")
data = spark.read.format("mongo").option("uri","mongodb://localhost:27017/music_database.audio_features_small").load()

# Flatten the arrays
data = data.withColumn("mfcc", flatten(data["mfcc"]))
data = data.withColumn("spectral_centroid", flatten(data["spectral_centroid"]))
data = data.withColumn("zero_crossing_rate", flatten(data["zero_crossing_rate"]))

# Explode the 'mfcc' array into separate columns
print("Processing the data in arrays...")

for i in range(20):  # Assuming 'mfcc' has 20 elements
    data = data.withColumn(f'mfcc_{i}', data['mfcc'].getItem(i))

# Extract the single element from 'spectral_centroid' and 'zero_crossing_rate'
data = data.withColumn('spectral_centroid', data['spectral_centroid'].getItem(0))
data = data.withColumn('zero_crossing_rate', data['zero_crossing_rate'].getItem(0))

# Now 'mfcc' is split into 'mfcc_0', 'mfcc_1', ..., 'mfcc_19'
mfcc_cols = [f'mfcc_{i}' for i in range(20)]

# Assemble the features
print("Assembling the features...")

assembler = VectorAssembler(inputCols=mfcc_cols + ['spectral_centroid', 'zero_crossing_rate'], outputCol="features")
features = assembler.transform(data)

# Standardize features
print("Standardizing features...")

scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=False)
scalerModel = scaler.fit(features)
scaled_data = scalerModel.transform(features)

# Collect data to use in PyTorch
features_list = np.array(scaled_data.select("scaled_features").rdd.map(lambda row: row[0]).collect())

# Define PyTorch dataset and loader
print("starting training data process...")
class AudioDataset(Dataset):
    def __init__(self, features):
        self.features = torch.tensor(features, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]

dataset = AudioDataset(features_list)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# Define and train the neural network
class MusicEmbeddingNet(nn.Module):
    def __init__(self, input_size, embedding_size=22):
        super(MusicEmbeddingNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, embedding_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model
input_size = features_list.shape[1]
model = MusicEmbeddingNet(input_size)

# Training setup
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [10, 20, 50]

for lr in learning_rates:
    for batch_size in batch_sizes:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = MusicEmbeddingNet(input_size)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(10):  # Reduced number of epochs for quick tuning
            for features in dataloader:
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, features)
                loss.backward()
                optimizer.step()
        
        # Evaluate model here or capture loss for comparison
        print(f"Finished training with lr: {lr}, batch size: {batch_size}, final loss: {loss.item()}")
    
# Create a function to get embeddings from the model
print("Getting embeddings...") 
def get_embeddings(dataloader, model):
    model.eval()  # Set model to evaluation mode
    embeddings = []
    with torch.no_grad():  # No need to track gradients for inference
        for features in dataloader:
            embedded_features = model(features)
            embeddings.extend(embedded_features.numpy())  # Convert tensors to numpy arrays and store
    return np.array(embeddings)

# Get embeddings for the entire dataset
song_embeddings = get_embeddings(dataloader, model)

embedding_dim = song_embeddings.shape[1]  # Number of dimensions for each embedding

# Create Annoy index
annoy_index = AnnoyIndex(embedding_dim, 'angular')  # 'angular' is one of the distance metrics supported by Annoy

# Add all embeddings to the Annoy index
for i, vector in enumerate(song_embeddings):
    annoy_index.add_item(i, vector)

# Build the index
annoy_index.build(10)  # 10 trees for the index, more trees give higher precision when querying

def get_similar_tracks(track_index, num_neighbors=5):
    similar_items = annoy_index.get_nns_by_item(track_index, num_neighbors + 1)  # Including itself
    return similar_items[1:]  # Remove the first item since it's the query item itself

# Example: Get 5 similar tracks for the first track in your dataset
similar_tracks = get_similar_tracks(0, 5)
print("Indices of similar tracks:", similar_tracks) 
