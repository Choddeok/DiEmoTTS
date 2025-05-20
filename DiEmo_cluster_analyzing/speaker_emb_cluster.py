import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import defaultdict, Counter

def get_emotion_label(audio_num):
    if audio_num <= 350:
        return "Neutral"
    elif audio_num <= 700:
        return "Angry"
    elif audio_num <= 1050:
        return "Happy"
    elif audio_num <= 1400:
        return "Sad"
    else:
        return "Surprise"

def load_embeddings(data_dir, filter_file):
    """
    Load embeddings from .pt files whose names are listed in filter_file.
    Returns dictionaries of embeddings, corresponding emotion labels, and original filenames.
    """
    embeddings = defaultdict(list)
    labels = defaultdict(list)
    # Additional: dictionary to store original .pt file names
    filenames = defaultdict(list)

    # Read filtered .pt file list
    with open(filter_file, 'r') as f:
        filtered_files = set(f.read().splitlines())

    for file in filtered_files:
        filepath = os.path.join(data_dir, file)
        if os.path.exists(filepath) and file.endswith(".pt"):
            tensor = torch.load(filepath).squeeze(0).numpy()  # [1, 192] -> [192]
            
            speaker_id, audio_num = file.split('_')[0], int(file.split('_')[1].split('.')[0])
            embeddings[speaker_id].append(tensor)
            labels[speaker_id].append(get_emotion_label(audio_num))
            # Additional: store the original file name
            filenames[speaker_id].append(file)
        else:
            print(f"Warning: File {file} listed in {filter_file} does not exist in {data_dir}")

    return embeddings, labels, filenames

def cluster_embeddings(method="kmeans", **kwargs):
    """Create a clustering model using the specified method."""
    if method == "kmeans":
        n_clusters = kwargs.get("n_clusters", 5)
        return KMeans(n_clusters=n_clusters, init="k-means++", n_init=50, random_state=kwargs.get("random_state", 42))
    elif method == "dbscan":
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        return DBSCAN(eps=eps, min_samples=min_samples)
    elif method == "spectral":
        n_clusters = kwargs.get("n_clusters", 5)
        return SpectralClustering(n_clusters=n_clusters, random_state=kwargs.get("random_state", 42))
    elif method == "gmm":
        n_clusters = kwargs.get("n_clusters", 5)
        return GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=kwargs.get("random_state", 42))
    else:
        raise ValueError(f"Unknown clustering method: {method}")

# Color mapping for true labels
real_label_colors = {
    "Neutral": "darkgrey",
    "Angry": "lightcoral",
    "Happy": "lightgreen",
    "Sad": "gold",
    "Surprise": "royalblue"  # corrected typo: 'royalbule' -> 'royalblue'
}

# Color mapping for cluster labels
cluster_label_colors = {
    "cluster0": "royalblue",
    "cluster1": "lightgreen",
    "cluster2": "gold",
    "cluster3": "lightcoral",
    "cluster4": "darkgrey",  # corrected typo: 'royalbule' -> 'royalblue'
}

def cluster_and_visualize(embeddings, labels, output_dir, clustering_method="kmeans", **kwargs):
    """
    Cluster embeddings for each speaker and visualize results with both true and predicted labels.
    """
    os.makedirs(output_dir, exist_ok=True)

    for speaker_id, speaker_embeddings in embeddings.items():
        # Reduce dimensions to 2D using PCA
        reduced_embeddings = PCA(n_components=2).fit_transform(speaker_embeddings)
        emotions = labels[speaker_id]
        
        # Perform clustering
        clusterer = cluster_embeddings(method=clustering_method, **kwargs)
        cluster_labels = clusterer.fit_predict(speaker_embeddings)

        # [1] Visualize true labels (emotions: Neutral, Angry, Happy, Sad, Surprise)
        plt.figure(figsize=(6, 6))
        for emotion in set(emotions):
            indices = [i for i, e in enumerate(emotions) if e == emotion]
            # Use color specified in real_label_colors (default to 'black' if not specified)
            color = real_label_colors.get(emotion, "black")
            plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1],
                        label=emotion, color=color, alpha=0.7)
        plt.xticks([])
        plt.yticks([])
        output_path_true = os.path.join(output_dir, f"{speaker_id}_true_labels_{clustering_method}.png")
        plt.savefig(output_path_true)
        plt.close()
        print(f"Saved true label visualization for speaker {speaker_id} at {output_path_true}")

        # [2] Visualize predicted cluster labels (e.g., cluster0, cluster1, etc.)
        plt.figure(figsize=(6, 6))
        for cluster in set(cluster_labels):
            indices = [i for i, c in enumerate(cluster_labels) if c == cluster]
            cluster_str = f"cluster{cluster}"
            # Use color specified in cluster_label_colors (default to 'black' if not specified)
            color = cluster_label_colors.get(cluster_str, "black")
            plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1],
                        label=cluster_str, color=color, alpha=0.7)
        plt.xticks([])
        plt.yticks([])
        output_path_pred = os.path.join(output_dir, f"{speaker_id}_predicted_clusters_{clustering_method}.png")
        plt.savefig(output_path_pred)
        plt.close()
        print(f"Saved predicted cluster visualization for speaker {speaker_id} at {output_path_pred}")

def calculate_purity(cluster_labels, true_labels):
    """Calculate the purity of the clustering."""
    cluster_label_counts = Counter(zip(cluster_labels, true_labels))
    purity = sum(max(cluster_label_counts[(c, t)] 
                     for t in set(true_labels) if (c, t) in cluster_label_counts) 
                 for c in set(cluster_labels))
    return purity / len(true_labels)

def evaluate_clustering(embeddings, labels, output_file, clustering_method="kmeans", **kwargs):
    """
    Cluster embeddings, evaluate against true labels, and save results to a text file.
    """
    metrics = {"ARI": 0, "NMI": 0, "Purity": 0}
    results = {}
    num_speakers = len(embeddings)

    with open(output_file, "w") as f:
        for speaker_id, speaker_embeddings in embeddings.items():
            true_labels = labels[speaker_id]
            clusterer = cluster_embeddings(method=clustering_method, **kwargs)
            cluster_labels = clusterer.fit_predict(speaker_embeddings)

            # Calculate metrics
            ari = adjusted_rand_score(true_labels, cluster_labels)
            nmi = normalized_mutual_info_score(true_labels, cluster_labels, average_method='arithmetic')
            purity = calculate_purity(cluster_labels, true_labels)

            metrics["ARI"] += ari
            metrics["NMI"] += nmi
            metrics["Purity"] += purity

            results[speaker_id] = {"ARI": ari, "NMI": nmi, "Purity": purity}
            f.write(f"Speaker {speaker_id} - ARI: {ari:.3f}, NMI: {nmi:.3f}, Purity: {purity:.3f}\n")

        # Write overall averages
        avg_metrics = {k: v / num_speakers for k, v in metrics.items()}
        f.write("\n=== Overall Averages ===\n")
        for metric, value in avg_metrics.items():
            f.write(f"Average {metric}: {value:.3f}\n")

    print(f"Results saved to {output_file}")
    return results

# =============================================
# Additional: Function to record the original .pt file names
# =============================================
def save_clustered_file_list(
    output_file, 
    speaker_embeddings, 
    speaker_labels, 
    speaker_filenames,  # Additional: file name dictionary
    clustering_method="kmeans", 
    **kwargs
):
    """
    Save a new text file containing the original .pt file name, original emotion, cluster label, and most frequent emotion.
    """
    with open(output_file, "w") as f:
        for speaker_id, embeddings in speaker_embeddings.items():
            true_labels = speaker_labels[speaker_id]

            # Perform clustering
            clusterer = cluster_embeddings(method=clustering_method, **kwargs)
            cluster_labels = clusterer.fit_predict(embeddings)

            # Determine the most frequent emotion in each cluster
            cluster_to_emotion = {}
            for cluster in set(cluster_labels):
                indices = [i for i, c in enumerate(cluster_labels) if c == cluster]
                emotions_in_cluster = [true_labels[i] for i in indices]
                most_common_emotion = Counter(emotions_in_cluster).most_common(1)[0][0]
                cluster_to_emotion[cluster] = most_common_emotion

            # Write data to file
            for idx, embedding in enumerate(embeddings):
                original_emotion = true_labels[idx]
                cluster_label = cluster_labels[idx]
                most_common_emotion = cluster_to_emotion[cluster_label]

                # Use the original .pt file name as is
                pt_file_name = speaker_filenames[speaker_id][idx]

                # Example output: "001_000123.pt,Angry,2,Angry"
                f.write(f"{pt_file_name},{original_emotion},{cluster_label},{most_common_emotion}\n")

# =============================================
# Execution Example
# =============================================

if __name__ == "__main__":
    data_directory = "/workspace/none/hd0/dataset/ECAPA_v2_emb"
    output_directory_kmeans = "./ECAPA_v2_emb_Clustering_Results_kmeans_list"
    output_file_kmeans = os.path.join(output_directory_kmeans, "clustering_metrics.txt")
    filter_file_path = "./filtered_pt_file_list.txt"

    # Create output folder
    os.makedirs(output_directory_kmeans, exist_ok=True)

    # Load embeddings and return filenames as well
    speaker_embeddings, speaker_labels, speaker_filenames = load_embeddings(data_directory, filter_file_path)

    # Perform clustering and visualization
    cluster_and_visualize(
        speaker_embeddings, 
        speaker_labels, 
        output_directory_kmeans, 
        clustering_method="kmeans", 
        n_clusters=5
    )

    # Evaluate clustering metrics
    evaluate_clustering(
        speaker_embeddings, 
        speaker_labels, 
        output_file_kmeans, 
        clustering_method="kmeans", 
        n_clusters=5
    )

    # Save final clustered file list with .pt file names, emotion, and cluster information
    output_txt_file2 = os.path.join(output_directory_kmeans, "final_clustered_file_list.txt")
    save_clustered_file_list(
        output_txt_file2, 
        speaker_embeddings, 
        speaker_labels, 
        speaker_filenames,  # Pass the dictionary of file names as well
        clustering_method="kmeans", 
        n_clusters=5
    )
