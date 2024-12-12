import os
import sys
import numpy as np
import logging
from datasets import load_dataset
from collections import Counter
from embedding import EmbeddingAnalyzer

import os
import subprocess

# Download data if not exists
if not os.path.exists("data/words_freq.csv"):
    os.system("curl -L -o data/words_freq.zip https://www.kaggle.com/api/v1/datasets/download/rtatman/english-word-frequency")
    os.system("unzip data/words_freq.zip -d data")
    os.system("mv data/unigram_freq.csv data/words_freq.csv")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load pkd/marxism-small
logging.info("Loading dataset pkd/marxism-small")
dataset = load_dataset("pkd/marxism-small")

def get_dist_matrix(words, model):
    distances = np.zeros((len(words), len(words)))

    # Compute the distance between each pair of words
    analyser = EmbeddingAnalyzer(model)
    logging.info("Computing distances between words")
    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            distances[i, j] = analyser.compute_distance(w1, w2)
            logging.info(f"Computed distance for word pair ({i}, {j})")
            logging.info(f"Distance: {distances[i, j]}")

    return distances

def relativize(frequencies: Counter):
    max_freq = frequencies.most_common(1)[0][1]

    rel_freqs = {}

    length = len(frequencies)
    i = 0
    for word, freq in frequencies.items():
        print(f"Relativizing {word} ({i}/{length})")
        rel_freqs[word] = freq / max_freq
        i += 1
    
    return rel_freqs


      
    

if __name__ == "__main__":
    n_words = 1000
    n_dists = 1000

    base_model = "gpt2"
    trained_model = "pkd/MarxGPT-2-aligned"

    # find 100 random english words

    words = []
    logging.info("Finding 100 random english words")
    with open("data/words_freq.csv") as f:
        lines = f.readlines()
        i = 0
        for line in lines[1:]:
            word = line.split(",")[0]
            words.append(word)
            if i >= n_words:
                break
            i += 1
    
    print(f"Found {len(words)} words")



    logging.info("Computing distance matrix for base model")
    distances_base = get_dist_matrix(words, base_model)
    np.save("data/distances_base.npy", distances_base)

    logging.info("Computing distance matrix for trained model")
    distances_trained = get_dist_matrix(words, trained_model)
    np.save("data/distances_trained.npy", distances_trained)



    diffs = distances_trained - distances_base
    np.save("data/diffs.npy", diffs)


    # write to a file
    np.save("data/diffs.npy", diffs)


    # Print the n_dists most changed distances
    logging.info(f"Printing the {n_dists} most changed distances")
    diffs_flat = diffs.flatten()

    idxs = np.argsort(diffs_flat)


    for i in range(int(n_dists / 2 )):
        # get half most negative and half most positive
        idx = idxs[i]
        i, j = idx // n_words, idx % n_words
        print(f"({words[i]}, {words[j]}): {diffs_flat[idx]}")
        print(f"\tBase distance: {distances_base[i, j]}")
        print(f"\tTrained distance: {distances_trained[i, j]}")
        print()

    # Print the n_dists least changed distances
    logging.info(f"Printing the {n_dists} least changed distances")
    for i in range(int(n_dists / 2)):
        # get half most negative and half most positive
        idx = idxs[-i - 1]
        i, j = idx // n_words, idx % n_words
        print(f"({words[i]}, {words[j]}): {diffs_flat[idx]}")
        print(f"\tBase distance: {distances_base[i, j]}")
        print(f"\tTrained distance: {distances_trained[i, j]}")
        print()

    # find the words which meaning changed the most
    logging.info("Finding the words which meaning changed the most")
    word_diffs = diffs.sum(axis=1)
    idxs = np.argsort(np.abs(word_diffs))[::-1]
    for i in range(n_words):
        print(f"{words[idxs[i]]}: {word_diffs[idxs[i]]}")

    # find the words which meaning changed the least
    logging.info("Finding the words which meaning changed the least")
    for i in range(n_words):
        print(f"{words[idxs[-i - 1]]}: {word_diffs[idxs[-i - 1]]}")

    import matplotlib.pyplot as plt

    # graph the 10 most changed words
    # use a bar graph with the difference as the height
    # legend with the word below the bar
    
    logging.info("Graphing the 10 most changed words")
    idxs = np.argsort(np.abs(word_diffs))[::-1]
    plt.bar(range(10), np.abs(word_diffs[idxs[:10]]))
    plt.xticks(range(10), [words[i] for i in idxs[:10]], rotation=55, fontsize=12)
    plt.show()