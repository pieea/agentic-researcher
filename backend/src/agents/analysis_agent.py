from sentence_transformers import SentenceTransformer
import hdbscan
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class AnalysisAgent:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            NumPy array of embeddings (shape: [n_texts, embedding_dim])
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 2
    ) -> np.ndarray:
        """
        Cluster embeddings using HDBSCAN with K-means fallback.

        Args:
            embeddings: NumPy array of embeddings
            min_cluster_size: Minimum cluster size for HDBSCAN

        Returns:
            NumPy array of cluster labels
        """
        logger.info(f"Clustering {len(embeddings)} embeddings")
        n_docs = len(embeddings)

        # Try HDBSCAN first with relaxed parameters
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(2, min(min_cluster_size, n_docs // 5)),
                min_samples=1,
                metric='euclidean',
                cluster_selection_method='eom'
            )

            labels = clusterer.fit_predict(embeddings)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_count = sum(labels == -1)

            logger.info(f"HDBSCAN found {n_clusters} clusters (noise: {noise_count}/{n_docs})")

            # If HDBSCAN found good clusters, use them
            if n_clusters >= 2 and noise_count < n_docs * 0.5:
                return labels

        except Exception as e:
            logger.warning(f"HDBSCAN failed: {str(e)}, falling back to K-means")

        # Fallback to K-means with dynamic cluster count
        # Use sqrt(n) as a heuristic for number of clusters
        n_clusters_kmeans = max(2, min(5, int(np.sqrt(n_docs))))
        logger.info(f"Using K-means with {n_clusters_kmeans} clusters")

        kmeans = KMeans(
            n_clusters=n_clusters_kmeans,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(embeddings)

        logger.info(f"K-means created {len(set(labels))} clusters")
        return labels

    def extract_cluster_keywords(
        self,
        texts: List[str],
        labels: np.ndarray,
        top_k: int = 5
    ) -> Dict[int, List[str]]:
        """
        Extract top keywords for each cluster using TF-IDF.

        Args:
            texts: Original text documents
            labels: Cluster labels for each document
            top_k: Number of top keywords to extract per cluster

        Returns:
            Dictionary mapping cluster_id -> list of keywords
        """
        cluster_keywords = {}
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:  # Skip noise
                continue

            # Get texts for this cluster
            cluster_texts = [text for i, text in enumerate(texts) if labels[i] == label]

            if not cluster_texts:
                continue

            # Extract keywords using TF-IDF
            try:
                vectorizer = TfidfVectorizer(max_features=top_k, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                keywords = vectorizer.get_feature_names_out()
                cluster_keywords[int(label)] = list(keywords)
            except:
                cluster_keywords[int(label)] = []

        return cluster_keywords
