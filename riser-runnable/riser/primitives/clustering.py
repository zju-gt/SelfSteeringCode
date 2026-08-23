from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import logging

from .extractor import ActivationPair

logger = logging.getLogger(__name__)


class PrimitiveClustering:
    
    def __init__(
        self,
        n_clusters: int = 6,
        pca_components: int = 2,
        pca_whiten: bool = True,
        kmeans_n_init: int = 10,
        kmeans_max_iter: int = 300,
        random_state: int = 42,
        representative_method: str = "centroid",
    ):
        try:
            from sklearn.decomposition import PCA
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ImportError(
                "PrimitiveClustering requires scikit-learn. "
                "Install the runnable dependencies with: pip install -e ."
            ) from exc

        self.n_clusters = n_clusters
        self.pca_components = pca_components
        self.representative_method = representative_method
        self.random_state = random_state
        
        self.pca = PCA(
            n_components=pca_components,
            whiten=pca_whiten,
            random_state=random_state,
        )
        
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=kmeans_n_init,
            max_iter=kmeans_max_iter,
            random_state=random_state,
        )
        
        self.scaler = StandardScaler()
        
        self.cluster_labels_ = None
        self.cluster_centers_ = None
        self.pca_projection_ = None
        
    def fit(
        self,
        activation_pairs: List[ActivationPair],
        quality_scores: Optional[List[float]] = None,
    ) -> Dict[int, torch.Tensor]:
        logger.info(f"Clustering {len(activation_pairs)} activation pairs into {self.n_clusters} primitives")
        
        differences = torch.stack([pair.difference for pair in activation_pairs])
        differences_np = differences.numpy()
        
        differences_scaled = self.scaler.fit_transform(differences_np)
        
        logger.info("Running KMeans clustering...")
        self.cluster_labels_ = self.kmeans.fit_predict(differences_scaled)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        
        logger.info("Computing PCA projection for visualization...")
        self.pca_projection_ = self.pca.fit_transform(differences_scaled)
        
        primitives = self._select_representatives(
            differences,
            differences_scaled,
            activation_pairs,
            quality_scores,
        )
        
        self._log_cluster_stats(activation_pairs)
        
        return primitives
    
    def _select_representatives(
        self,
        differences: torch.Tensor,
        differences_scaled: np.ndarray,
        activation_pairs: List[ActivationPair],
        quality_scores: Optional[List[float]] = None,
    ) -> Dict[int, torch.Tensor]:
        primitives = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels_ == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                logger.warning(f"Cluster {cluster_id} is empty!")
                continue
            
            if self.representative_method == "centroid":
                centroid_scaled = self.cluster_centers_[cluster_id]
                centroid = self.scaler.inverse_transform(centroid_scaled.reshape(1, -1))
                representative = torch.from_numpy(centroid[0]).float()
                
            elif self.representative_method == "medoid":
                centroid = self.cluster_centers_[cluster_id]
                cluster_points = differences_scaled[cluster_mask]
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                medoid_idx = cluster_indices[np.argmin(distances)]
                representative = differences[medoid_idx]
                
            elif self.representative_method == "best_quality":
                if quality_scores is None:
                    logger.warning(
                        "best_quality method requires quality_scores, falling back to medoid"
                    )
                    centroid = self.cluster_centers_[cluster_id]
                    cluster_points = differences_scaled[cluster_mask]
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    medoid_idx = cluster_indices[np.argmin(distances)]
                    representative = differences[medoid_idx]
                else:
                    cluster_scores = [quality_scores[i] for i in cluster_indices]
                    best_idx = cluster_indices[np.argmax(cluster_scores)]
                    representative = differences[best_idx]
            else:
                raise ValueError(f"Unknown representative method: {self.representative_method}")
            
            norm = representative.norm()
            if norm > 0:
                representative = representative / norm
            primitives[cluster_id] = representative
            logger.debug(
                f"Cluster {cluster_id}: {len(cluster_indices)} samples, "
                f"representative shape: {representative.shape}"
            )
        
        return primitives
    
    def _log_cluster_stats(self, activation_pairs: List[ActivationPair]):
        from collections import Counter
        
        cluster_counts = Counter(self.cluster_labels_)
        logger.info("Cluster distribution:")
        for cluster_id in range(self.n_clusters):
            count = cluster_counts.get(cluster_id, 0)
            percentage = 100 * count / len(activation_pairs)
            logger.info(f"  Cluster {cluster_id}: {count} samples ({percentage:.1f}%)")
        
        logger.info("Sample tasks per cluster:")
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels_ == cluster_id
            cluster_pairs = [p for i, p in enumerate(activation_pairs) if cluster_mask[i]]
            
            if cluster_pairs:
                tasks = [p.task for p in cluster_pairs[:3]]
                logger.info(f"  Cluster {cluster_id}: {tasks}")
    
    def visualize(
        self,
        save_path: Optional[str] = None,
        primitive_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 8),
    ):
        import matplotlib.pyplot as plt

        if self.pca_projection_ is None:
            raise ValueError("Must call fit() before visualize()")
        
        if self.pca_components != 2:
            logger.warning(f"PCA has {self.pca_components} components, using first 2 for visualization")
        
        plt.figure(figsize=figsize)
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels_ == cluster_id
            cluster_points = self.pca_projection_[cluster_mask]
            
            label = (
                primitive_names[cluster_id]
                if primitive_names and cluster_id < len(primitive_names)
                else f"Cluster {cluster_id}"
            )
            
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                label=label,
                alpha=0.6,
                s=50,
            )
        
        if self.cluster_centers_ is not None:
            centers_pca = self.pca.transform(self.cluster_centers_)
            plt.scatter(
                centers_pca[:, 0],
                centers_pca[:, 1],
                c='black',
                marker='X',
                s=200,
                linewidths=2,
                edgecolors='white',
                label='Centroids',
                zorder=10,
            )
        
        plt.xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Latent Space Visualization of Cognitive Primitives')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def get_cluster_assignments(self) -> Dict[int, List[int]]:
        if self.cluster_labels_ is None:
            raise ValueError("Must call fit() before get_cluster_assignments()")
        
        assignments = {}
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels_ == cluster_id
            assignments[cluster_id] = np.where(cluster_mask)[0].tolist()
        
        return assignments
