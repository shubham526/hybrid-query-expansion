"""
Memory-efficient version of MultiVectorReranker that avoids pre-encoding all documents.
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import gc

logger = logging.getLogger(__name__)


class MemoryEfficientMultiVectorReranker:
    """
    Memory-efficient multi-vector reranking system with on-demand document encoding.
    """

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 device: str = None,
                 max_query_vectors: int = 64,
                 doc_encoding_batch_size: int = 8,
                 enable_doc_cache: bool = True,
                 max_doc_cache_size: int = 1000):
        """
        Initialize memory-efficient multi-vector reranking system.

        Args:
            model_name: Sentence transformer model name
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            max_query_vectors: Maximum number of vectors in query representation
            doc_encoding_batch_size: Batch size for document encoding
            enable_doc_cache: Whether to cache document encodings
            max_doc_cache_size: Maximum number of documents to cache
        """
        self.model_name = model_name
        self.max_query_vectors = max_query_vectors
        self.doc_encoding_batch_size = doc_encoding_batch_size
        self.enable_doc_cache = enable_doc_cache
        self.max_doc_cache_size = max_doc_cache_size

        # Load sentence transformer model
        try:
            self.model = SentenceTransformer(model_name, device=device)
            self.tokenizer = self.model.tokenizer
            self.device = self.model.device
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

        # Document encoding cache (LRU-style)
        if self.enable_doc_cache:
            self._doc_cache = {}
            self._doc_cache_order = []
        else:
            self._doc_cache = None

        # Token embedding cache for query construction
        self._token_embedding_cache = {}

        logger.info(f"Memory-efficient reranker initialized with {model_name}")
        logger.info(f"Document encoding batch size: {doc_encoding_batch_size}")
        logger.info(f"Document cache: {'enabled' if enable_doc_cache else 'disabled'}")

    def _get_document_from_cache(self, doc_id: str) -> Optional[torch.Tensor]:
        """Get document encoding from cache if available."""
        if not self.enable_doc_cache or doc_id not in self._doc_cache:
            return None

        # Move to end (most recently used)
        self._doc_cache_order.remove(doc_id)
        self._doc_cache_order.append(doc_id)

        return self._doc_cache[doc_id]

    def _store_document_in_cache(self, doc_id: str, encoding: torch.Tensor):
        """Store document encoding in cache with LRU eviction."""
        if not self.enable_doc_cache:
            return

        # Remove oldest entries if cache is full
        while len(self._doc_cache) >= self.max_doc_cache_size:
            oldest_doc_id = self._doc_cache_order.pop(0)
            del self._doc_cache[oldest_doc_id]

        # Add new encoding
        self._doc_cache[doc_id] = encoding
        self._doc_cache_order.append(doc_id)

    def encode_documents_batch(self, documents: List[Tuple[str, str]],
                               max_length: int = 512) -> Dict[str, torch.Tensor]:
        """
        Encode a batch of documents efficiently.

        Args:
            documents: List of (doc_id, doc_text) tuples
            max_length: Maximum sequence length

        Returns:
            Dictionary mapping doc_id -> encoded vectors
        """
        batch_encodings = {}

        # Separate cached and uncached documents
        to_encode = []
        for doc_id, doc_text in documents:
            cached_encoding = self._get_document_from_cache(doc_id)
            if cached_encoding is not None:
                batch_encodings[doc_id] = cached_encoding
            else:
                to_encode.append((doc_id, doc_text))

        if not to_encode:
            return batch_encodings

        logger.debug(f"Encoding {len(to_encode)} documents (batch size: {self.doc_encoding_batch_size})")

        # Process in smaller batches
        for i in range(0, len(to_encode), self.doc_encoding_batch_size):
            batch = to_encode[i:i + self.doc_encoding_batch_size]

            try:
                # Encode batch of documents
                doc_texts = [doc_text for _, doc_text in batch]
                doc_ids = [doc_id for doc_id, _ in batch]

                # Batch encode all documents at once for efficiency
                with torch.no_grad():
                    # Use sentence-level encoding for documents (simpler approach)
                    doc_embeddings = self.model.encode(
                        doc_texts,
                        convert_to_tensor=True,
                        batch_size=self.doc_encoding_batch_size,
                        show_progress_bar=False
                    )

                # Store results
                for doc_id, embedding in zip(doc_ids, doc_embeddings):
                    # Ensure embedding has correct shape [num_vectors, embedding_dim]
                    if embedding.dim() == 1:
                        embedding = embedding.unsqueeze(0)

                    batch_encodings[doc_id] = embedding
                    self._store_document_in_cache(doc_id, embedding)

            except Exception as e:
                logger.warning(f"Error encoding document batch: {e}")
                # Fallback: encode individually
                for doc_id, doc_text in batch:
                    try:
                        encoding = self._encode_single_document(doc_text, max_length)
                        batch_encodings[doc_id] = encoding
                        self._store_document_in_cache(doc_id, encoding)
                    except Exception as e2:
                        logger.warning(f"Failed to encode document {doc_id}: {e2}")
                        # Create dummy encoding
                        embedding_dim = self.model.get_sentence_embedding_dimension()
                        batch_encodings[doc_id] = torch.zeros(1, embedding_dim, device=self.device)

        return batch_encodings

    def _encode_single_document(self, document: str, max_length: int = 512) -> torch.Tensor:
        """Encode a single document into multi-vector representation."""
        try:
            if not document or not document.strip():
                # Return single zero vector for empty documents
                embedding_dim = self.model.get_sentence_embedding_dimension()
                return torch.zeros(1, embedding_dim, device=self.device)

            # For memory efficiency, use sentence-level encoding
            # You can switch back to token-level if needed
            with torch.no_grad():
                embedding = self.model.encode([document], convert_to_tensor=True)[0]

            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)

            return embedding

        except Exception as e:
            logger.warning(f"Error encoding document: {e}")
            embedding_dim = self.model.get_sentence_embedding_dimension()
            return torch.zeros(1, embedding_dim, device=self.device)

    def create_importance_weighted_query_vectors(self,
                                                 query: str,
                                                 expansion_terms: List[Tuple[str, float]],
                                                 importance_weights: Dict[str, float]) -> torch.Tensor:
        """Create importance-weighted query vectors (same as before)."""
        # Keep the same implementation as the original
        query_vectors = []

        # Original query tokens
        try:
            query_tokens = self.tokenizer.tokenize(query)
            for token in query_tokens[:self.max_query_vectors // 2]:
                embedding = self._get_token_embedding(token)
                if embedding is not None:
                    query_vectors.append(embedding)
        except Exception as e:
            logger.warning(f"Error tokenizing query '{query}': {e}")
            query_embedding = self.model.encode([query], convert_to_tensor=True)[0]
            query_vectors.append(query_embedding)

        # Expansion terms with importance scaling
        for term, rm_weight in expansion_terms:
            if len(query_vectors) >= self.max_query_vectors:
                break

            importance_score = importance_weights.get(term, 0.0)
            if importance_score <= 0:
                continue

            try:
                subword_tokens = self.tokenizer.tokenize(term)
                for subword in subword_tokens:
                    if len(query_vectors) >= self.max_query_vectors:
                        break

                    embedding = self._get_token_embedding(subword)
                    if embedding is not None:
                        scaled_embedding = importance_score * embedding
                        query_vectors.append(scaled_embedding)
            except Exception as e:
                logger.debug(f"Error processing expansion term '{term}': {e}")
                continue

        if not query_vectors:
            query_embedding = self.model.encode([query], convert_to_tensor=True)[0]
            query_vectors = [query_embedding]

        return torch.stack(query_vectors)

    def _get_token_embedding(self, token: str) -> Optional[torch.Tensor]:
        """Get token embedding (same as before)."""
        if token in self._token_embedding_cache:
            return self._token_embedding_cache[token]

        try:
            token_id = self.tokenizer.convert_tokens_to_ids([token])[0]
            if token_id != self.tokenizer.unk_token_id:
                embedding = None
                try:
                    if hasattr(self.model, '_modules') and '0' in self.model._modules:
                        auto_model = self.model[0].auto_model
                        if hasattr(auto_model, 'embeddings'):
                            embedding = auto_model.embeddings.word_embeddings.weight[token_id].detach()
                except (AttributeError, IndexError, KeyError):
                    pass

                if embedding is None:
                    try:
                        embedding = self.model.encode([token], convert_to_tensor=True)[0]
                    except Exception:
                        pass

                if embedding is not None:
                    self._token_embedding_cache[token] = embedding
                    return embedding

            return None
        except Exception as e:
            logger.debug(f"Error getting embedding for token '{token}': {e}")
            return None

    def late_interaction_score(self, query_vectors: torch.Tensor,
                               doc_vectors: torch.Tensor) -> float:
        """Compute late interaction score (same as before)."""
        if query_vectors.numel() == 0 or doc_vectors.numel() == 0:
            return 0.0

        try:
            if query_vectors.device != doc_vectors.device:
                doc_vectors = doc_vectors.to(query_vectors.device)

            if query_vectors.dim() == 1:
                query_vectors = query_vectors.unsqueeze(0)
            if doc_vectors.dim() == 1:
                doc_vectors = doc_vectors.unsqueeze(0)

            similarities = torch.matmul(query_vectors, doc_vectors.transpose(0, 1))
            max_sims, _ = torch.max(similarities, dim=1)
            total_score = torch.sum(max_sims).item()

            return total_score

        except Exception as e:
            logger.warning(f"Error computing late interaction score: {e}")
            return 0.0

    def rerank_streaming(self,
                         query: str,
                         expansion_terms: List[Tuple[str, float]],
                         importance_weights: Dict[str, float],
                         candidate_results: List[Tuple[str, str, float]],
                         top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Memory-efficient reranking with streaming document encoding.

        Args:
            query: Original query text
            expansion_terms: List of (term, rm_weight) tuples
            importance_weights: Dictionary mapping terms to importance scores
            candidate_results: List of (doc_id, doc_text, first_stage_score)
            top_k: Number of top results to return

        Returns:
            List of (doc_id, reranking_score) tuples
        """
        if not candidate_results:
            return []

        logger.info(f"Memory-efficient reranking of {len(candidate_results)} candidates")

        # Create query vectors once
        try:
            query_vectors = self.create_importance_weighted_query_vectors(
                query, expansion_terms, importance_weights
            )
        except Exception as e:
            logger.error(f"Failed to create query vectors: {e}")
            return [(doc_id, first_stage_score) for doc_id, _, first_stage_score in
                    sorted(candidate_results, key=lambda x: x[2], reverse=True)[:top_k]]

        reranked_scores = []

        # Process documents in batches
        batch_size = self.doc_encoding_batch_size
        num_batches = (len(candidate_results) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(candidate_results))
            batch_candidates = candidate_results[start_idx:end_idx]

            logger.debug(f"Processing batch {batch_idx + 1}/{num_batches} "
                         f"({len(batch_candidates)} documents)")

            # Prepare batch for encoding
            batch_docs = [(doc_id, doc_text) for doc_id, doc_text, _ in batch_candidates]

            # Encode batch of documents
            try:
                doc_encodings = self.encode_documents_batch(batch_docs)

                # Score each document in the batch
                for doc_id, doc_text, first_stage_score in batch_candidates:
                    try:
                        if doc_id in doc_encodings:
                            doc_vectors = doc_encodings[doc_id]
                            reranking_score = self.late_interaction_score(query_vectors, doc_vectors)
                            reranked_scores.append((doc_id, reranking_score))
                        else:
                            # Fallback to first-stage score
                            reranked_scores.append((doc_id, first_stage_score))
                    except Exception as e:
                        logger.warning(f"Error scoring document {doc_id}: {e}")
                        reranked_scores.append((doc_id, first_stage_score))

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                # Fallback: use first-stage scores for this batch
                for doc_id, _, first_stage_score in batch_candidates:
                    reranked_scores.append((doc_id, first_stage_score))

            # Clear GPU memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Optional: clear document encodings to save memory
            # (they're cached separately if caching is enabled)
            del doc_encodings
            gc.collect()

        # Sort by reranking score and return top k
        reranked_scores.sort(key=lambda x: x[1], reverse=True)
        return reranked_scores[:top_k]

    def get_memory_stats(self) -> Dict[str, Union[str, int]]:
        """Get memory usage statistics."""
        stats = {
            'doc_cache_size': len(self._doc_cache) if self._doc_cache else 0,
            'max_doc_cache_size': self.max_doc_cache_size,
            'token_cache_size': len(self._token_embedding_cache),
            'doc_encoding_batch_size': self.doc_encoding_batch_size,
            'cache_enabled': self.enable_doc_cache
        }

        if torch.cuda.is_available():
            stats['gpu_memory_allocated'] = f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB"
            stats['gpu_memory_reserved'] = f"{torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB"

        return stats

    def clear_caches(self):
        """Clear all caches to free memory."""
        if self._doc_cache:
            self._doc_cache.clear()
            self._doc_cache_order.clear()

        self._token_embedding_cache.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
        logger.info("Cleared all caches")


# Wrapper function to maintain compatibility with existing code
def create_memory_efficient_reranker(model_name: str = 'all-MiniLM-L6-v2',
                                     large_candidate_sets: bool = True) -> MemoryEfficientMultiVectorReranker:
    """
    Factory function to create memory-efficient reranker with appropriate settings.

    Args:
        model_name: Sentence transformer model name
        large_candidate_sets: Whether you expect large candidate sets

    Returns:
        Configured memory-efficient reranker
    """
    if large_candidate_sets:
        # Conservative settings for large candidate sets
        return MemoryEfficientMultiVectorReranker(
            model_name=model_name,
            doc_encoding_batch_size=4,  # Smaller batches
            enable_doc_cache=True,
            max_doc_cache_size=500  # Smaller cache
        )
    else:
        # More aggressive settings for smaller candidate sets
        return MemoryEfficientMultiVectorReranker(
            model_name=model_name,
            doc_encoding_batch_size=16,  # Larger batches
            enable_doc_cache=True,
            max_doc_cache_size=2000  # Larger cache
        )


# Example usage showing memory monitoring
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Create memory-efficient reranker
    reranker = create_memory_efficient_reranker('all-MiniLM-L6-v2', large_candidate_sets=True)

    # Example data
    query = "machine learning algorithms"
    expansion_terms = [("neural", 0.8), ("networks", 0.6), ("classification", 0.5)]
    importance_weights = {"neural": 1.5, "networks": 1.2, "classification": 0.8}

    # Simulate large candidate set
    candidate_results = [
        (f"doc_{i}", f"Document {i} about machine learning and neural networks", 0.9 - i * 0.01)
        for i in range(1000)  # 1000 candidates
    ]

    print("Memory stats before reranking:")
    print(reranker.get_memory_stats())

    # Rerank with streaming
    results = reranker.rerank_streaming(
        query=query,
        expansion_terms=expansion_terms,
        importance_weights=importance_weights,
        candidate_results=candidate_results,
        top_k=10
    )

    print(f"\nTop 5 results:")
    for i, (doc_id, score) in enumerate(results[:5], 1):
        print(f"{i}. {doc_id}: {score:.4f}")

    print("\nMemory stats after reranking:")
    print(reranker.get_memory_stats())