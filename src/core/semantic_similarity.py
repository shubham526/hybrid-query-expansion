"""
Semantic Similarity Module

Clean semantic similarity computation using sentence transformers.
Optimized for query expansion in IR tasks.

Author: Your Name
"""

import logging
import numpy as np
import torch
from typing import List, Union, Dict
from functools import lru_cache

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SemanticSimilarity:
    """
    Semantic similarity computation using sentence transformers.
    Optimized for query expansion use cases.
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 device: str = None,
                 cache_size: int = 1000):
        """
        Initialize semantic similarity computer.

        Args:
            model_name: Sentence transformer model name
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            cache_size: Size of embedding cache (0 to disable)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.device = self.model.device

        # Setup caching if enabled
        if cache_size > 0:
            self._encode_single = lru_cache(maxsize=cache_size)(self._encode_single_uncached)
        else:
            self._encode_single = self._encode_single_uncached

        logger.info(f"Loaded sentence transformer: {model_name} on {self.device}")

    def _encode_single_uncached(self, text: str) -> np.ndarray:
        """Encode single text without caching."""
        with torch.no_grad():
            embedding = self.model.encode([text], convert_to_tensor=False)[0]
        return embedding

    def encode(self,
               texts: Union[str, List[str]],
               batch_size: int = 32) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode text(s) into embeddings.

        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for processing multiple texts

        Returns:
            Single embedding array or list of embedding arrays
        """
        if isinstance(texts, str):
            return self._encode_single(texts)
        else:
            with torch.no_grad():
                embeddings = self.model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_tensor=False
                )
            return embeddings

    def compute_similarity(self,
                          text1: Union[str, np.ndarray],
                          text2: Union[str, np.ndarray]) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text (string or pre-computed embedding)
            text2: Second text (string or pre-computed embedding)

        Returns:
            Cosine similarity score (0 to 1)
        """
        # Get embeddings
        if isinstance(text1, str):
            emb1 = self._encode_single(text1)
        else:
            emb1 = text1

        if isinstance(text2, str):
            emb2 = self._encode_single(text2)
        else:
            emb2 = text2

        # Compute cosine similarity
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = np.dot(emb1, emb2) / (norm1 * norm2)
        return max(0.0, float(similarity))  # Ensure non-negative

    def compute_query_expansion_similarities(self,
                                           query: str,
                                           expansion_terms: List[str]) -> Dict[str, float]:
        """
        Compute similarities between query and expansion terms.
        Optimized for query expansion use case.

        Args:
            query: Original query
            expansion_terms: List of expansion terms

        Returns:
            Dictionary mapping terms to similarity scores
        """
        if not expansion_terms:
            return {}

        # Encode query once
        query_emb = self._encode_single(query)

        # Compute similarities for all terms
        similarities = {}
        for term in expansion_terms:
            term_emb = self._encode_single(term)
            similarity = self.compute_similarity(query_emb, term_emb)
            similarities[term] = similarity

        return similarities

    def clear_cache(self):
        """Clear the embedding cache."""
        if hasattr(self._encode_single, 'cache_clear'):
            self._encode_single.cache_clear()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize
    sim_computer = SemanticSimilarity('all-MiniLM-L6-v2')

    # Example usage
    query = "machine learning algorithms"
    expansion_terms = ["neural", "networks", "supervised", "classification"]

    # Compute similarities
    similarities = sim_computer.compute_query_expansion_similarities(query, expansion_terms)

    print(f"Query: {query}")
    print("Expansion term similarities:")
    for term, sim in similarities.items():
        print(f"  {term:<15} {sim:.4f}")