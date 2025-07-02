#!/usr/bin/env python3
"""
Unit tests for RM expansion module.

Tests the core functionality of RM1 and RM3 query expansion including:
- Basic expansion functionality
- Term weighting computation
- Score normalization
- Edge cases and error handling
- Different RM types (RM1 vs RM3)

Usage:
    python -m pytest test_rm_expansion.py -v
    python test_rm_expansion.py  # Run directly
"""

import unittest
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import tempfile
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.rm_expansion import RMExpansion, rm1_expansion, rm3_expansion


class TestRMExpansion(unittest.TestCase):
    """Test cases for RM expansion functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.rm_expansion = RMExpansion()

        # Sample documents for testing
        self.sample_documents = [
            "Machine learning algorithms are used in artificial intelligence applications.",
            "Neural networks and deep learning models process natural language text.",
            "Information retrieval systems use algorithms to find relevant documents.",
            "Computer science research focuses on machine learning and data mining.",
            "Artificial intelligence and machine learning transform modern technology."
        ]

        # Sample scores (pseudo-relevance scores)
        self.sample_scores = [0.9, 0.8, 0.7, 0.6, 0.5]

        # Sample query
        self.sample_query = "machine learning algorithms"

    def test_basic_rm3_expansion(self):
        """Test basic RM3 expansion functionality."""
        expansion_terms = self.rm_expansion.expand_query(
            query=self.sample_query,
            documents=self.sample_documents,
            scores=self.sample_scores,
            num_expansion_terms=10,
            rm_type="rm3"
        )

        # Should return list of (term, weight) tuples
        self.assertIsInstance(expansion_terms, list)
        self.assertGreater(len(expansion_terms), 0)

        # Each item should be a tuple with term and weight
        for term, weight in expansion_terms:
            self.assertIsInstance(term, str)
            self.assertIsInstance(weight, float)
            self.assertGreater(weight, 0.0)

        # Should include original query terms (RM3)
        expansion_words = [term for term, weight in expansion_terms]
        self.assertIn("machine", expansion_words)
        self.assertIn("learning", expansion_words)

    def test_basic_rm1_expansion(self):
        """Test basic RM1 expansion functionality."""
        expansion_terms = self.rm_expansion.expand_query(
            query=self.sample_query,
            documents=self.sample_documents,
            scores=self.sample_scores,
            num_expansion_terms=10,
            rm_type="rm1"
        )

        # Should return expansion terms
        self.assertIsInstance(expansion_terms, list)
        self.assertGreater(len(expansion_terms), 0)

        # Should NOT include original query terms (RM1)
        expansion_words = [term for term, weight in expansion_terms]

        # Original query terms should be filtered out in RM1
        query_terms = set(self.sample_query.lower().split())
        expansion_set = set(expansion_words)

        # There might be some overlap, but expansion should contain new terms
        self.assertGreater(len(expansion_set - query_terms), 0)

    def test_convenience_functions(self):
        """Test RM1 and RM3 convenience functions."""
        # Test rm1_expansion function
        rm1_terms = rm1_expansion(
            query=self.sample_query,
            documents=self.sample_documents,
            scores=self.sample_scores,
            num_terms=5
        )

        self.assertEqual(len(rm1_terms), 5)

        # Test rm3_expansion function
        rm3_terms = rm3_expansion(
            query=self.sample_query,
            documents=self.sample_documents,
            scores=self.sample_scores,
            num_terms=5
        )

        self.assertEqual(len(rm3_terms), 5)

        # RM3 should generally have higher weights for query terms
        rm3_words = {term: weight for term, weight in rm3_terms}

        # Check if query terms are present and have reasonable weights
        for query_term in ["machine", "learning"]:
            if query_term in rm3_words:
                self.assertGreater(rm3_words[query_term], 0.0)

    def test_term_filtering(self):
        """Test term filtering functionality."""
        # Test with custom stopwords
        custom_stopwords = {"machine", "learning", "algorithms"}
        rm_expansion = RMExpansion(
            stopwords=custom_stopwords,
            min_term_length=3,
            max_term_length=15
        )

        expansion_terms = rm_expansion.expand_query(
            query=self.sample_query,
            documents=self.sample_documents,
            scores=self.sample_scores,
            num_expansion_terms=10,
            rm_type="rm1"
        )

        # Filtered terms should not appear in expansion
        expansion_words = [term for term, weight in expansion_terms]
        for stopword in custom_stopwords:
            self.assertNotIn(stopword, expansion_words)

        # All terms should meet length requirements
        for term, weight in expansion_terms:
            self.assertGreaterEqual(len(term), 3)
            self.assertLessEqual(len(term), 15)

    def test_score_normalization(self):
        """Test score normalization functionality."""
        # Test with different score ranges
        high_scores = [10.0, 9.0, 8.0, 7.0, 6.0]
        low_scores = [0.1, 0.09, 0.08, 0.07, 0.06]

        # Both should produce valid expansion terms
        expansion_high = self.rm_expansion.expand_query(
            query=self.sample_query,
            documents=self.sample_documents,
            scores=high_scores,
            num_expansion_terms=5,
            rm_type="rm3"
        )

        expansion_low = self.rm_expansion.expand_query(
            query=self.sample_query,
            documents=self.sample_documents,
            scores=low_scores,
            num_expansion_terms=5,
            rm_type="rm3"
        )

        # Both should return valid results
        self.assertEqual(len(expansion_high), 5)
        self.assertEqual(len(expansion_low), 5)

        # Weights should be normalized (roughly between 0 and 1)
        for term, weight in expansion_high:
            self.assertGreater(weight, 0.0)
            self.assertLess(weight, 2.0)  # Allow some flexibility

        for term, weight in expansion_low:
            self.assertGreater(weight, 0.0)
            self.assertLess(weight, 2.0)

    def test_empty_input_handling(self):
        """Test handling of empty inputs."""
        # Empty documents should return an empty list
        expansion_empty_docs = self.rm_expansion.expand_query(
            query=self.sample_query,
            documents=[],
            scores=[],
            num_expansion_terms=5,
            rm_type="rm3"
        )
        self.assertEqual(len(expansion_empty_docs), 0)

        # Empty query should still run and return a list (possibly empty)
        expansion_empty_query = self.rm_expansion.expand_query(
            query="",
            documents=self.sample_documents,
            scores=self.sample_scores,
            num_expansion_terms=5,
            rm_type="rm3"
        )
        self.assertIsInstance(expansion_empty_query, list)

        # Mismatched documents and scores should raise a ValueError
        # This is the corrected part of the test.
        with self.assertRaises(ValueError):
            self.rm_expansion.expand_query(
                query=self.sample_query,
                documents=self.sample_documents,  # 5 documents
                scores=[0.9, 0.8],  # 2 scores
                num_expansion_terms=5,
                rm_type="rm3"
            )

    def test_weight_computation_consistency(self):
        """Test that weight computation is consistent and deterministic."""
        # Run expansion multiple times with same input
        expansion1 = self.rm_expansion.expand_query(
            query=self.sample_query,
            documents=self.sample_documents,
            scores=self.sample_scores,
            num_expansion_terms=10,
            rm_type="rm3"
        )

        expansion2 = self.rm_expansion.expand_query(
            query=self.sample_query,
            documents=self.sample_documents,
            scores=self.sample_scores,
            num_expansion_terms=10,
            rm_type="rm3"
        )

        # Results should be identical
        self.assertEqual(expansion1, expansion2)

        # Weights should be sorted in descending order
        weights = [weight for term, weight in expansion1]
        self.assertEqual(weights, sorted(weights, reverse=True))

    def test_different_document_content(self):
        """Test expansion with different types of document content."""
        # Test with single-word documents
        single_word_docs = ["machine", "learning", "algorithm", "neural", "network"]
        single_word_scores = [1.0, 0.9, 0.8, 0.7, 0.6]

        expansion_single = self.rm_expansion.expand_query(
            query="machine learning",
            documents=single_word_docs,
            scores=single_word_scores,
            num_expansion_terms=3,
            rm_type="rm3"
        )

        self.assertGreater(len(expansion_single), 0)

        # Test with very long documents
        long_docs = [
            " ".join(["artificial intelligence machine learning"] * 100),
            " ".join(["neural networks deep learning"] * 100)
        ]
        long_scores = [1.0, 0.8]

        expansion_long = self.rm_expansion.expand_query(
            query="artificial intelligence",
            documents=long_docs,
            scores=long_scores,
            num_expansion_terms=5,
            rm_type="rm3"
        )

        self.assertGreater(len(expansion_long), 0)

    def test_statistical_properties(self):
        """Test statistical properties of the expansion."""
        expansion_terms = self.rm_expansion.expand_query(
            query=self.sample_query,
            documents=self.sample_documents,
            scores=self.sample_scores,
            num_expansion_terms=10,
            rm_type="rm3"
        )

        # Compute basic statistics
        weights = [weight for term, weight in expansion_terms]

        # All weights should be positive
        self.assertTrue(all(w > 0 for w in weights))

        # Weights should sum to a reasonable value (depends on implementation)
        total_weight = sum(weights)
        self.assertGreater(total_weight, 0.0)

        # Top terms should have higher weights than bottom terms
        if len(weights) > 1:
            self.assertGreater(weights[0], weights[-1])

    def test_edge_cases(self):
        """Test various edge cases."""
        # Very small number of expansion terms
        small_expansion = self.rm_expansion.expand_query(
            query=self.sample_query,
            documents=self.sample_documents,
            scores=self.sample_scores,
            num_expansion_terms=1,
            rm_type="rm3"
        )
        self.assertEqual(len(small_expansion), 1)

        # Very large number of expansion terms (more than available)
        large_expansion = self.rm_expansion.expand_query(
            query=self.sample_query,
            documents=self.sample_documents,
            scores=self.sample_scores,
            num_expansion_terms=1000,
            rm_type="rm3"
        )
        # Should return available terms, not fail
        self.assertGreater(len(large_expansion), 0)
        self.assertLess(len(large_expansion), 1000)

        # Zero scores
        zero_scores = [0.0] * len(self.sample_documents)
        zero_expansion = self.rm_expansion.expand_query(
            query=self.sample_query,
            documents=self.sample_documents,
            scores=zero_scores,
            num_expansion_terms=5,
            rm_type="rm3"
        )
        # Should handle gracefully (might return empty or uniform weights)
        self.assertIsInstance(zero_expansion, list)

    def test_custom_configuration(self):
        """Test RM expansion with custom configuration."""
        custom_rm = RMExpansion(
            stopwords={"the", "and", "or"},
            min_term_length=4,
            max_term_length=12,
            remove_query_terms=True
        )

        expansion_terms = custom_rm.expand_query(
            query=self.sample_query,
            documents=self.sample_documents,
            scores=self.sample_scores,
            num_expansion_terms=8,
            rm_type="rm3"
        )

        # Check configuration is applied
        expansion_words = [term for term, weight in expansion_terms]

        # Terms should meet length requirements
        for term in expansion_words:
            self.assertGreaterEqual(len(term), 4)
            self.assertLessEqual(len(term), 12)

        # Should not contain specified stopwords
        for stopword in ["the", "and", "or"]:
            self.assertNotIn(stopword, expansion_words)


class TestRMExpansionIntegration(unittest.TestCase):
    """Integration tests for RM expansion with realistic data."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.rm_expansion = RMExpansion()

        # More realistic documents (similar to MSMARCO passages)
        self.realistic_documents = [
            "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
            "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data.",
            "Information retrieval is the activity of obtaining information system resources that are relevant to an information need from a collection of those resources. Searches can be based on full-text or other content-based indexing.",
            "Artificial intelligence is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals, which involves consciousness and emotionality."
        ]

    def test_realistic_query_expansion(self):
        """Test expansion with realistic query and documents."""
        queries = [
            "machine learning algorithms",
            "natural language processing",
            "artificial intelligence systems",
            "information retrieval methods"
        ]

        scores = [0.95, 0.87, 0.78, 0.69, 0.62]

        for query in queries:
            with self.subTest(query=query):
                # Test RM3 expansion
                rm3_terms = self.rm_expansion.expand_query(
                    query=query,
                    documents=self.realistic_documents,
                    scores=scores,
                    num_expansion_terms=10,
                    rm_type="rm3"
                )

                # Should return reasonable expansion
                self.assertGreater(len(rm3_terms), 0)
                self.assertLessEqual(len(rm3_terms), 10)

                # Check that we get meaningful terms
                expansion_words = [term for term, weight in rm3_terms]

                # Should contain some relevant terms
                relevant_terms = ["learning", "intelligence", "data", "systems", "methods", "analysis"]
                found_relevant = any(term in expansion_words for term in relevant_terms)
                self.assertTrue(found_relevant, f"No relevant terms found for query: {query}")

    def test_performance_characteristics(self):
        """Test performance characteristics of RM expansion."""
        import time

        # Test with larger document set
        large_documents = self.realistic_documents * 20  # 100 documents
        large_scores = [0.9 - i * 0.001 for i in range(len(large_documents))]

        start_time = time.time()
        expansion_terms = self.rm_expansion.expand_query(
            query="machine learning artificial intelligence",
            documents=large_documents,
            scores=large_scores,
            num_expansion_terms=15,
            rm_type="rm3"
        )
        end_time = time.time()

        # Should complete in reasonable time (< 5 seconds for 100 docs)
        elapsed = end_time - start_time
        self.assertLess(elapsed, 5.0, f"RM expansion took too long: {elapsed:.2f}s")

        # Should still return valid results
        self.assertEqual(len(expansion_terms), 15)


def run_all_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestRMExpansion))
    suite.addTests(loader.loadTestsFromTestCase(TestRMExpansionIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    # Run tests when script is executed directly
    import argparse

    parser = argparse.ArgumentParser(description="Test RM expansion functionality")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--test', '-t', type=str, default=None,
                        help='Run specific test method')

    args = parser.parse_args()

    if args.test:
        # Run specific test
        suite = unittest.TestSuite()
        suite.addTest(TestRMExpansion(args.test))
        runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
        result = runner.run(suite)
    else:
        # Run all tests
        success = run_all_tests()
        if not success:
            sys.exit(1)

    print("\nAll tests completed!")