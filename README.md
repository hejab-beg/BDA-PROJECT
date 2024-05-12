# Project: Developing a Spotify Alternative
## Overview
Spotify is a digital music streaming service offering millions of songs, podcasts, and videos from global artists. Users can enjoy music for free with ads or opt for a premium plan for an ad-free experience, offline listening, and high-quality audio. Spotify's recommendation system employs machine learning algorithms to analyze user behavior and generate personalized recommendations, considering factors like listening habits, playlists, and followed artists. This project aims to develop a streamlined alternative to Spotify, featuring a music recommendation system, playback, and streaming capabilities, alongside real-time suggestions derived from user activity.
## Extract, Transform, Load (ETL) 

### Audio compression:
- As the dataset was too large so we had to use audio compression to reduce its size and make it managable.
- in the code we used FFmpeg to compress audio files in a specified folder while retaining metadata. It iterates through each MP3 file, applies compression with a target bitrate (default: 64 kbps), and replaces the original file with the compressed one.
-  The script provides a simple and efficient way to reduce file sizes while preserving audio quality ad its metadata.
  
### ETL:
- Extracred audio features such as MFCC, spectral centroid, and zero-crossing rate from a dataset of MP3 files using libraries like librosa and sklearn.
- Standardized and normalized the features for consistency and applies Principal Component Analysis (PCA) to reduce the dimensionality of MFCC features while preserving variance.
- The processed features are then asynchronously inserted into a MongoDB collection using pymongo and concurrent.futures.ThreadPoolExecutor. Error handling and logging ensure robustness, while parallel processing enhances efficiency.
- This comprehensive approach enables efficient extraction, processing, and storage of audio features for further analysis and application.

## Music Recommendation Model

- Implemented a music recommendation system using PySpark for data preprocessing and PyTorch for training a neural network to generate embeddings.
- Key steps include data loading from MongoDB, feature engineering, model training, and building an Annoy index for nearest neighbor search.
- The trained model learns embeddings for music tracks, facilitating recommendation based on track similarity.
- The system's flexibility allows tuning hyperparameters such as learning rate and batch size for optimal performance.
- Finally, Annoy's efficient indexing enables fast retrieval of similar tracks.

## Deployment
- Now in deployment, music streaming web application was developed, focusing on user interaction and seamless recommendation integration.
- Utilizing frameworks using spark, the application will offer a well-structured and user-friendly interface.
-  Apache Kafka was leveraged to dynamically generate real-time music recommendations, incorporating historical playback data for personalized suggestions.
-  The application abstains from user-uploaded audio files, relying solely on Apache Kafka for recommendation generation based on user activity, thus ensures a seamless and tailored streaming experience.

  ## Team members
  - Zunaira Ahmed (22I-2075)
  - Hejab beg (22I-2071)
  - Sara Zahid (22I-1861)
