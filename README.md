# 📘 Clustering Assignment – Complete Submission

This repository contains all required clustering implementations for the CMPE / Data Mining Clustering Assignment.
Each clustering method is implemented in its own Google Colab notebook with explanations, visualizations, and evaluation metrics.

The notebooks demonstrate clustering on synthetic, real, document, image, audio, and time-series data, using both classical ML and modern embedding-based techniques.


## 🗂️ Repository Structure
Clustering_Assignment/
│── Audio_clustering_using_ImageBind_audio_e....ipynb
│── Dbscan_clustering_using_pycaret.ipynb
│── Document_clustering_with_LLM_embeddings_1.ipynb
│── Gaussian_mixture_models_clustering_.ipynb
│── Hierarchical_clustering_.ipynb
│── Illustrate_clustering_.ipynb
│── Image_clustering_using_ImageBind_embed....ipynb
│── K_Means_clustering_from_scratch.ipynb
│── anomaly_detection_using_pyOD_.ipynb
│── README.md


### ✅ Assignment Requirements & Notebook Links

Below is the official list (a → i) matched with your notebooks.


a) K-Means Clustering From Scratch

📄 Notebook: K_Means_clustering_from_scratch.ipynb

Contents:

Manual implementation of:

1. Distance computation
2. Cluster assignment
3. Centroid updates
4. Convergence criteria
5. Visualization of clustering
6. Comparison with sklearn's KMeans
7. Silhouette score evaluation

References:
https://colab.sandbox.google.com/github/SANTOSHMAHER/Machine-Learning-Algorithms/blob/master/K_Means_algorithm_using_Python_from_scratch_ipynb
https://colab.sandbox.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.11-KMeans.ipynb

b) Hierarchical Clustering (Not from Scratch)

📄 Notebook: Hierarchical_clustering_.ipynb

Contents:

1. Agglomerative clustering
2. Dendrograms using SciPy
3. Different linkage methods
4. Cluster visualization

🔗 Reference:
https://colab.sandbox.google.com/github/saskeli/data-analysis-with-python-summer-2019/blob/master/clustering.ipynb

c) Gaussian Mixture Models (Not from Scratch)

📄 Notebook: Gaussian_mixture_models_clustering_.ipynb

Contents:

1. EM algorithm via scikit-learn
2. Soft assignments
3. AIC / BIC comparison
4. Visualization of mixture components

🔗 Reference:
https://colab.sandbox.google.com/github/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/05.12-Gaussian-Mixtures.ipynb

d) DBSCAN Clustering using PyCaret

📄 Notebook: Dbscan_clustering_using_pycaret.ipynb

Contents:

1. PyCaret clustering setup
2. DBSCAN model creation
3. Label assignment
4. PyCaret auto-plots:
   t-SNE
   Cluster overlay
   Feature distribution
5. Metrics extraction

🔗 References:
https://pycaret.org/create-model/
https://towardsdatascience.com/clustering-made-easy-with-pycaret-656316c0b080

e) Anomaly Detection with PyOD

📄 Notebook: anomaly_detection_using_pyOD_.ipynb

Contents:

1. Univariate and multivariate outlier detection
2. Algorithms used:
    Isolation Forest
    Local Outlier Factor
    Autoencoder (optional)
3. Visualization of anomalies
4. Decision boundaries

🔗 Reference:
https://neptune.ai/blog/anomaly-detection-in-time-series

f) Time-Series Clustering using Pretrained Models

📄 Notebook: Illustrate_clustering_.ipynb

Contents:

1. Time-series embedding extraction (e.g., TS2Vec / pretrained encoders)
2. Dimensionality reduction (UMAP/PCA)
3. K-Means clustering
4. Time-series comparison plots

🔗 Useful Links:
https://github.com/V-MalM/Stock-Clustering-and-Prediction
https://github.com/qianlima-lab/time-series-ptms
https://github.com/effa/time-series-clustering

g) Document Clustering using LLM Embeddings

📄 Notebook: Document_clustering_with_LLM_embeddings_1.ipynb

Contents:

1. SentenceTransformer (all-MiniLM) embeddings
2. K-Means clustering
3. Silhouette score
4. UMAP visualization
5. Label assignment for each document

🔗 Helpful sources:

https://github.com/simonnw/llm-cluster
https://simonwillison.net/2023/Sep/4/llm-embeddings/
https://github.com/UKPLab/sentence-transformers

h) Image Clustering using ImageBind Embeddings

📄 Notebook:
Image_clustering_using_ImageBind_embed....ipynb

Contents:

1. Image preprocessing
2. ImageBind visual embedding extraction
3. K-Means / DBSCAN clustering
4. Cluster visualization
5. Silhouette score

🔗 References:

https://medium.com/@tatsuroomurata317/image-bind-metai-on-google-colab-free-843f3004977c
https://towardsdatascience.com/introduction-to-embedding-clustering-and-similarity-11dd80b00061

i) Audio Clustering using ImageBind Embeddings

📄 Notebook:
Audio_clustering_using_ImageBind_audio_e....ipynb

Contents:

1. Audio file ingestion & preprocessing
2. Audio embeddings via ImageBind
3. K-Means clustering
4. Embedding shape validation
5. Cluster interpretation

🔗 References:

https://mct-master.github.io/machine-learning/2023/04/25/ninjak-clustering-audio.html
https://ridkhan5.medium.com/audio-clustering-with-deep-learning-a7991d605fa5


### 📊 Clustering Metrics Used

Across notebooks:

Silhouette Score

Inertia (K-Means)

AIC / BIC (GMM)

Cluster visual separation (UMAP/t-SNE)

DBSCAN density analysis

Outlier score distributions (PyOD)


#### 📚 Datasets

Datasets include:

Synthetic datasets generated via scikit-learn

Open-source datasets from:

Kaggle

PapersWithCode: https://paperswithcode.com/datasets

Public time-series repositories

Custom small text, image, and audio samples for embeddings
