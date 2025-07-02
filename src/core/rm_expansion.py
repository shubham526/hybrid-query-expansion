"""
RM (Relevance Model) Expansion Module

Implements RM1 and RM3 query expansion algorithms based on Lavrenko & Croft (2001).
Provides clean, reusable interface for pseudo-relevance feedback query expansion.

Author: Your Name
"""

import re
import logging
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional, Set
from math import log, exp

try:
    import nltk
    from nltk.corpus import stopwords

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)


class RMExpansion:
    """
    Relevance Model (RM) query expansion implementation.

    Supports both RM1 (expansion terms only) and RM3 (original query + expansion terms).
    Based on Lavrenko & Croft (2001) relevance-based language models.
    """

    def __init__(self,
                 stopwords: Optional[Set[str]] = None,
                 min_term_length: int = 2,
                 max_term_length: int = 20,
                 remove_query_terms: bool = False,
                 use_nltk_stopwords: bool = True,
                 language: str = 'english'):
        """
        Initialize RM expansion with filtering parameters.

        Args:
            stopwords: Set of stopwords to filter out (None to use NLTK or fallback)
            min_term_length: Minimum length for expansion terms
            max_term_length: Maximum length for expansion terms
            remove_query_terms: Whether to remove original query terms from expansion
            use_nltk_stopwords: Whether to use NLTK stopwords (if available)
            language: Language for NLTK stopwords (default: 'english')
        """
        self.min_term_length = min_term_length
        self.max_term_length = max_term_length
        self.remove_query_terms = remove_query_terms
        self.language = language

        # Initialize stopwords
        if stopwords is not None:
            self.stopwords = stopwords
        elif use_nltk_stopwords and NLTK_AVAILABLE:
            self.stopwords = self._get_nltk_stopwords(language)
        else:
            self.stopwords = self._get_fallback_stopwords()
            if use_nltk_stopwords and not NLTK_AVAILABLE:
                logger.warning("NLTK not available, using fallback stopwords. Install with: pip install nltk")

        logger.debug(f"Initialized RMExpansion with {len(self.stopwords)} stopwords")

    def _get_nltk_stopwords(self, language: str = 'english') -> Set[str]:
        """
        Get NLTK stopwords for specified language.

        Args:
            language: Language for stopwords (default: 'english')

        Returns:
            Set of stopwords
        """
        try:
            # Try to get stopwords, download if necessary
            try:
                stop_words = set(stopwords.words(language))
            except LookupError:
                logger.info(f"Downloading NLTK stopwords for {language}...")
                nltk.download('stopwords', quiet=True)
                stop_words = set(stopwords.words(language))

            logger.debug(f"Loaded {len(stop_words)} NLTK stopwords for {language}")
            return stop_words

        except Exception as e:
            logger.warning(f"Failed to load NLTK stopwords: {e}. Using fallback stopwords.")
            return self._get_fallback_stopwords()

    def _get_fallback_stopwords(self) -> Set[str]:
        """Get fallback English stopwords if NLTK is not available."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'this',
            'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'can', 'may', 'might', 'must', 'shall', 'ought'
        }

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using simple regex-based approach.

        Args:
            text: Input text to tokenize

        Returns:
            List of filtered tokens
        """
        if not text or not isinstance(text, str):
            return []

        # Basic tokenization: extract alphabetic words
        tokens = re.findall(r'\b[a-zA-Z]+\b', text.lower())

        # Apply filters
        filtered_tokens = []
        for token in tokens:
            if (self.min_term_length <= len(token) <= self.max_term_length
                    and token not in self.stopwords):
                filtered_tokens.append(token)

        return filtered_tokens

    def expand_query(self,
                     query: str,
                     documents: List[str],
                     scores: List[float],
                     num_expansion_terms: int = 10,
                     rm_type: str = "rm3") -> List[Tuple[str, float]]:
        """
        Expand query using RM1 or RM3 algorithm.

        Args:
            query: Original query string
            documents: List of pseudo-relevant documents
            scores: Relevance scores for each document
            num_expansion_terms: Number of expansion terms to return
            rm_type: "rm1" (expansion only) or "rm3" (query + expansion)

        Returns:
            List of (term, weight) tuples, sorted by weight (descending)

        Raises:
            ValueError: If documents and scores have different lengths
            ValueError: If rm_type is not "rm1" or "rm3"
        """
        if not documents or not scores:
            logger.warning("No documents or scores provided for expansion")
            return []

        if len(documents) != len(scores):
            raise ValueError(f"Documents ({len(documents)}) and scores ({len(scores)}) must have same length")

        if rm_type not in ["rm1", "rm3"]:
            raise ValueError(f"rm_type must be 'rm1' or 'rm3', got '{rm_type}'")

        logger.debug(f"Expanding query '{query}' using {rm_type.upper()} with {len(documents)} documents")

        # Tokenize query
        query_terms = self.tokenize(query)
        logger.debug(f"Query terms: {query_terms}")

        # Compute term weights using relevance model
        term_weights = self._compute_relevance_model(
            documents=documents,
            scores=scores,
            query_terms=query_terms if rm_type == "rm3" else None
        )

        # Remove query terms if requested
        if self.remove_query_terms:
            for query_term in query_terms:
                term_weights.pop(query_term, None)

        # Sort by weight and return top terms
        sorted_terms = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)
        result = sorted_terms[:num_expansion_terms]

        logger.debug(f"Expansion completed: {len(result)} terms")
        return result

    def _compute_relevance_model(self,
                                 documents: List[str],
                                 scores: List[float],
                                 query_terms: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Compute relevance model term weights.

        Args:
            documents: List of pseudo-relevant documents
            scores: Document relevance scores
            query_terms: Original query terms (for RM3, None for RM1)

        Returns:
            Dictionary mapping terms to their relevance model weights
        """
        # Normalize document scores to probabilities
        doc_probs = self._normalize_scores_to_probabilities(scores)

        # Initialize term weights
        term_weights = defaultdict(float)

        # Add original query terms with weight 1.0 (for RM3)
        if query_terms:
            for term in query_terms:
                if term not in self.stopwords:
                    term_weights[term] += 1.0

        # Process each document
        for doc_text, doc_prob in zip(documents, doc_probs):
            if doc_prob <= 0:
                continue

            # Tokenize document
            doc_tokens = self.tokenize(doc_text)
            if not doc_tokens:
                continue

            # Compute term frequencies in this document
            term_counts = Counter(doc_tokens)
            doc_length = len(doc_tokens)

            # Add weighted term frequencies to relevance model
            for term, count in term_counts.items():
                # P(term|doc) using maximum likelihood estimation
                term_prob_in_doc = count / doc_length

                # Weight by document probability: P(doc) * P(term|doc)
                term_weights[term] += doc_prob * term_prob_in_doc

        return dict(term_weights)

    def _normalize_scores_to_probabilities(self, scores: List[float]) -> List[float]:
        """
        Normalize relevance scores to probability distribution.

        Args:
            scores: List of relevance scores

        Returns:
            List of normalized probabilities
        """
        if not scores:
            return []

        # Check if scores are log probabilities (negative values)
        use_log_scores = any(score < 0 for score in scores)

        if use_log_scores:
            # Convert log scores to probabilities
            max_score = max(scores)
            probs = [exp(score - max_score) for score in scores]
            normalizer = sum(probs)

            if normalizer > 0:
                return [p / normalizer for p in probs]
            else:
                # Fallback to uniform distribution
                return [1.0 / len(scores)] * len(scores)
        else:
            # Regular scores - normalize to sum to 1
            total_score = sum(scores)

            if total_score > 0:
                return [score / total_score for score in scores]
            else:
                # Fallback to uniform distribution
                return [1.0 / len(scores)] * len(scores)

    def get_expansion_statistics(self,
                                 expansion_terms: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Compute statistics about expansion terms.

        Args:
            expansion_terms: List of (term, weight) tuples

        Returns:
            Dictionary with statistics (mean_weight, std_weight, etc.)
        """
        if not expansion_terms:
            return {}

        weights = [weight for _, weight in expansion_terms]

        import statistics
        stats = {
            'num_terms': len(expansion_terms),
            'mean_weight': statistics.mean(weights),
            'median_weight': statistics.median(weights),
            'min_weight': min(weights),
            'max_weight': max(weights)
        }

        if len(weights) > 1:
            stats['std_weight'] = statistics.stdev(weights)
        else:
            stats['std_weight'] = 0.0

        return stats

    def set_stopwords_from_nltk(self, language: str = 'english'):
        """
        Update stopwords using NLTK for specified language.

        Args:
            language: Language for NLTK stopwords
        """
        if NLTK_AVAILABLE:
            self.stopwords = self._get_nltk_stopwords(language)
            self.language = language
        else:
            logger.warning("NLTK not available. Install with: pip install nltk")

    def set_stopwords(self, stopwords: Set[str]):
        """
        Update the stopwords list.

        Args:
            stopwords: New set of stopwords
        """
        self.stopwords = stopwords
        logger.debug(f"Updated stopwords list to {len(stopwords)} words")

    def add_stopwords(self, additional_stopwords: Set[str]):
        """
        Add additional stopwords to the existing list.

        Args:
            additional_stopwords: Additional stopwords to add
        """
        self.stopwords.update(additional_stopwords)
        logger.debug(f"Added {len(additional_stopwords)} stopwords, total: {len(self.stopwords)}")


def get_available_stopword_languages() -> List[str]:
    """
    Get list of available languages for NLTK stopwords.

    Returns:
        List of language codes, empty list if NLTK not available
    """
    if not NLTK_AVAILABLE:
        return []

    try:
        # Try to get the list, download if necessary
        try:
            languages = stopwords.fileids()
        except LookupError:
            nltk.download('stopwords', quiet=True)
            languages = stopwords.fileids()

        return sorted(languages)
    except Exception as e:
        logger.warning(f"Failed to get NLTK stopword languages: {e}")
        return []


# Convenience functions for common usage patterns
def rm1_expansion(query: str,
                  documents: List[str],
                  scores: List[float],
                  num_terms: int = 10,
                  **kwargs) -> List[Tuple[str, float]]:
    """
    Convenience function for RM1 expansion.

    Args:
        query: Query string
        documents: Pseudo-relevant documents
        scores: Document scores
        num_terms: Number of expansion terms
        **kwargs: Additional arguments for RMExpansion

    Returns:
        List of (term, weight) tuples
    """
    rm = RMExpansion(**kwargs)
    return rm.expand_query(query, documents, scores, num_terms, rm_type="rm1")


def rm3_expansion(query: str,
                  documents: List[str],
                  scores: List[float],
                  num_terms: int = 10,
                  **kwargs) -> List[Tuple[str, float]]:
    """
    Convenience function for RM3 expansion.

    Args:
        query: Query string
        documents: Pseudo-relevant documents
        scores: Document scores
        num_terms: Number of expansion terms
        **kwargs: Additional arguments for RMExpansion

    Returns:
        List of (term, weight) tuples
    """
    rm = RMExpansion(**kwargs)
    return rm.expand_query(query, documents, scores, num_terms, rm_type="rm3")


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for example
    logging.basicConfig(level=logging.INFO)

    # Check NLTK availability
    print("NLTK Stopwords Integration Example")
    print("=" * 40)
    print(f"NLTK available: {NLTK_AVAILABLE}")

    if NLTK_AVAILABLE:
        available_languages = get_available_stopword_languages()
        print(f"Available stopword languages: {available_languages[:10]}...")  # Show first 10
    print()

    # Example usage
    query = "machine learning algorithms"

    # Mock pseudo-relevant documents
    documents = [
        "Machine learning algorithms include supervised and unsupervised methods. Neural networks and decision trees are popular algorithms.",
        "Deep learning neural networks have revolutionized machine learning. Convolutional networks excel at computer vision tasks.",
        "Support vector machines and random forests are classical machine learning methods for classification tasks.",
        "Supervised learning algorithms require labeled training data for classification and regression problems.",
        "Unsupervised learning discovers hidden patterns in data without labels using clustering algorithms."
    ]

    # Mock relevance scores (higher = more relevant)
    scores = [0.95, 0.87, 0.82, 0.78, 0.71]

    print(f"Query: {query}")
    print(f"Documents: {len(documents)}")
    print()

    # Initialize RM expansion with NLTK stopwords
    rm = RMExpansion(use_nltk_stopwords=True)
    print(f"Using {len(rm.stopwords)} stopwords")

    # Show some example stopwords
    sample_stopwords = sorted(list(rm.stopwords))[:20]
    print(f"Sample stopwords: {sample_stopwords}")
    print()

    # RM3 expansion
    print("RM3 Expansion (query + expansion terms):")
    rm3_terms = rm.expand_query(query, documents, scores, num_expansion_terms=10, rm_type="rm3")
    for i, (term, weight) in enumerate(rm3_terms, 1):
        print(f"{i:2d}. {term:<15} {weight:.4f}")
    print()

    # RM1 expansion
    print("RM1 Expansion (expansion terms only):")
    rm1_terms = rm.expand_query(query, documents, scores, num_expansion_terms=8, rm_type="rm1")
    for i, (term, weight) in enumerate(rm1_terms, 1):
        print(f"{i:2d}. {term:<15} {weight:.4f}")
    print()

    # Test different language (if available)
    if NLTK_AVAILABLE and 'spanish' in get_available_stopword_languages():
        print("Testing Spanish stopwords:")
        rm_spanish = RMExpansion(use_nltk_stopwords=True, language='spanish')
        print(f"Spanish stopwords count: {len(rm_spanish.stopwords)}")
        spanish_sample = sorted(list(rm_spanish.stopwords))[:10]
        print(f"Sample Spanish stopwords: {spanish_sample}")
        print()

    # Statistics
    print("Expansion Statistics:")
    stats = rm.get_expansion_statistics(rm3_terms)
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")