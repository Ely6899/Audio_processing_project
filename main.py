from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
from sklearn.decomposition import PCA

from Models.Encoder.inference import load_model, embed_utterance, preprocess_wav
from Models.Encoder.visualizations import Visualizations
from GeneralUtils.DirPaths import LIBRISPEECH_TRAIN_ROOT_FOLDER
from GeneralUtils.FileUtils import get_intermediate_folders
from Models.Encoder import *
from numpy.linalg import norm

import torch


encoder_model_path = Path("Models/Encoder/saved_models/train1.pt")
audio_hello = Path("Ely_Hello_World.flac")
audio_test = Path("Ely_Test.flac")
audio_book = Path("61-70968-0000.flac")

if __name__ == '__main__':
    #visualizer = Visualizations(disabled=True)
    load_model(encoder_model_path, "cuda")
    embedding_test = embed_utterance(preprocess_wav("Ely_Hello_World.flac", None)).reshape(1,-1)
    #embedding_test = embed_utterance(preprocess_wav(audio_test, None)).reshape(1,-1)
    #embedding_book = embed_utterance(preprocess_wav(audio_book, None)).reshape(1,-1)

    #np.save(Path("syntesizer_test.npy"),embedding_test)

    print(np.load(Path("syntesizer_test.npy")))

    # vectors = np.vstack([embedding_hello, embedding_test, embedding_book])  # Shape (3, 256)
    #
    # # Compute pairwise cosine similarities
    # cosine_similarities = np.dot(vectors, vectors.T) / (norm(vectors, axis=1)[:, np.newaxis] * norm(vectors, axis=1))
    #
    # # Convert cosine similarity to cosine distance (1 - similarity)
    # cosine_distances = 1 - cosine_similarities
    # print("Pairwise Cosine distances:\n", cosine_distances)

    # embeddings = np.hstack((embedding_hello, embedding_test, embedding_book)).T
    #
    # #visualizer.draw_projections(embeddings, utterances_per_speaker=2, step=1, out_fpath="embeds.png", max_speakers=1)
    # # Perform PCA to reduce to 2 dimensions
    # pca = PCA(n_components=2)
    # reduced_data = pca.fit_transform(embeddings)
    #
    # # Colors and labels for each point
    # colors = ['red', 'blue', 'green']
    # labels = ['hello', 'test', 'book']
    #
    # # Plot the results
    # plt.figure(figsize=(8, 6))
    # for i in range(len(labels)):
    #     plt.scatter(reduced_data[i, 0], reduced_data[i, 1], color=colors[i], label=labels[i])
    #
    # plt.title('Dimensionality Reduction with PCA')
    # plt.xlabel('Component 1')
    # plt.ylabel('Component 2')
    # plt.legend()
    # plt.grid(True)
    #
    # # Save the plot to disk
    # save_path = "reduced_plot_colored_points.png"
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.close()  # Close the plot to free memory

