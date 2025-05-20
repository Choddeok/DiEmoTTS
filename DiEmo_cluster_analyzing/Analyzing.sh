#!/bin/bash
set -e

##########################################
#   Clustering and Matching of Emotion   #
##########################################

# 0. Speaker Embeddings (ECAPA) Clustering
# Reference:
# I. R. Ulgen, Z. Du, C. Busso, and B. Sisman, “Revealing emotional clusters in speaker embeddings:
# A contrastive learning strategy for speech emotion recognition,” in IEEE Int. Conf. Acoust., Speech Signal Process. IEEE, 2024, pp. 12 081–12 085.
# python speaker_emb_cluster.py

# 1. Extract Neutral Emotion Cluster using Convex Hull Algorithm
# Reference:
# C. B. Barber, D. P. Dobkin, and H. Huhdanpaa, “The quickhull algorithm for convex hulls,” ACM Trans. Math. Softw.,
# vol. 22, no. 4, pp. 469–483, 1996.
python VAD_cluster_avg.py
python convex.py

# 2. Spherical Coordinate Transformation
# Reference:
# D.-H. Cho, H.-S. Oh, S.-B. Kim, S.-H. Lee, and S.-W. Lee, “Emosphere-tts: Emotional style and intensity modeling via spherical
# emotion vector for controllable emotional text-to-speech,” in Interspeech, 2024, pp. 1810–1814.
python spherical_transformer.py

# 3. Multiple Hungarian Methods for the k-Assignment Problem
# Reference:
# B. Gabrovšek, T. Novak, J. Povh, D. Rupnik Poklukar, and J. Žerovnik, “Multiple hungarian method for k-assignment problem,”
# Mathematics, vol. 8, no. 11, p. 2050, 2020.
python Hungarian_matching.py
