"""
Weight Optimizer Module

Learns optimal importance weights for combining RM, BM25, and semantic similarity scores
in query expansion. Uses direct metric optimization (L-BFGS-B) to maximize retrieval performance.

Author: Your Name
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from scipy.optimize import minimize
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class WeightOptimizer(ABC):
    """
    Abstract base class for weight optimization algorithms.
    """

    @abstractmethod
    def optimize_weights(self,
                         training_data: Optional[Dict],
                         validation_queries: Dict[str, str],
                         validation_qrels: Dict[str, Dict[str, int]],
                         evaluation_function: Callable[[Tuple[float, float, float]], float]) -> Tuple[float, float, float]:
        """
        Learn optimal weights for importance combination.

        Args:
            training_data: Training dataset with expansion features (can be None)
            validation_queries: query_id -> query_text (not used by evaluation_function)
            validation_qrels: query_id -> {doc_id: relevance} (not used by evaluation_function)
            evaluation_function: Function that takes (alpha, beta, gamma) and returns metric score

        Returns:
            Optimal (alpha, beta, gamma) weights for (RM, BM25, semantic)
        """
        pass


class LBFGSOptimizer(WeightOptimizer):
    """
    L-BFGS-B optimizer for learning expansion weights.
    Directly optimizes retrieval metrics (nDCG, MAP, etc.) on validation set.
    """

    def __init__(self,
                 bounds: List[Tuple[float, float]] = None,
                 max_iterations: int = 50,
                 tolerance: float = 1e-4):
        """
        Initialize L-BFGS-B optimizer.

        Args:
            bounds: List of (min, max) bounds for each weight
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance
        """
        self.bounds = bounds or [(0.1, 5.0), (0.1, 5.0), (0.1, 5.0)]
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.iterations = 0  # Track iterations for logging

        logger.info(f"L-BFGS-B optimizer initialized with bounds {self.bounds}")

    def optimize_weights(self,
                         training_data: Optional[Dict],
                         validation_queries: Dict[str, str],
                         validation_qrels: Dict[str, Dict[str, int]],
                         evaluation_function: Callable[[Tuple[float, float, float]], float]) -> Tuple[float, float, float]:
        """
        Learn optimal weights using L-BFGS-B optimization.

        Args:
            training_data: Training dataset (not used directly)
            validation_queries: Validation queries (not used directly - evaluation_function handles this)
            validation_qrels: Validation relevance judgments (not used directly)
            evaluation_function: Function that takes (alpha, beta, gamma) and returns metric score

        Returns:
            Optimal (α, β, γ) weights
        """
        self.iterations = 0
        iteration_scores = []

        def objective(weights):
            """Objective function: negative metric (since we minimize)."""
            alpha, beta, gamma = weights

            # Ensure positive weights
            if alpha < 0 or beta < 0 or gamma < 0:
                return 1.0  # High penalty for negative weights

            try:
                # Convert to tuple as expected by evaluation function
                weights_tuple = (alpha, beta, gamma)

                # Evaluate retrieval performance with these weights
                metric_score = evaluation_function(weights_tuple)

                self.iterations += 1
                iteration_scores.append((weights_tuple, metric_score))

                if self.iterations % 5 == 0:
                    logger.debug(f"Iteration {self.iterations}: weights={weights_tuple}, score={metric_score:.4f}")

                return -metric_score  # Negative because scipy minimizes
            except Exception as e:
                logger.warning(f"Error in objective function with weights {weights}: {e}")
                return 1.0  # High penalty for errors

        logger.info("Starting L-BFGS-B optimization...")

        # Evaluate baseline (equal weights)
        try:
            baseline_score = evaluation_function((1.0, 1.0, 1.0))
            logger.info(f"Baseline performance (equal weights): {baseline_score:.4f}")
        except Exception as e:
            logger.warning(f"Could not evaluate baseline: {e}")
            baseline_score = 0.0

        # Run optimization
        try:
            result = minimize(
                objective,
                x0=[1.0, 1.0, 1.0],  # Start with equal weights
                bounds=self.bounds,  # Weight bounds
                method='L-BFGS-B',
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.tolerance,
                    'disp': False
                }
            )

            optimal_weights = tuple(result.x)
            optimal_score = -result.fun

            logger.info("L-BFGS-B optimization completed!")
            logger.info(f"Optimal weights: alpha={optimal_weights[0]:.3f}, beta={optimal_weights[1]:.3f}, gamma={optimal_weights[2]:.3f}")
            logger.info(f"Optimal score: {optimal_score:.4f}")
            logger.info(f"Improvement: {optimal_score - baseline_score:+.4f}")
            logger.info(f"Convergence: {result.success}, Iterations: {result.nit}")

            # Store iterations for external access
            self.iterations = result.nit

            return optimal_weights

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            logger.info("Returning baseline weights (1.0, 1.0, 1.0)")
            return (1.0, 1.0, 1.0)


class GridSearchOptimizer(WeightOptimizer):
    """
    Grid search optimizer for learning expansion weights.
    Exhaustively searches over a grid of weight combinations.
    """

    def __init__(self,
                 weight_ranges: List[List[float]] = None,
                 resolution: int = 10):
        """
        Initialize grid search optimizer.

        Args:
            weight_ranges: List of [min, max] ranges for each weight
            resolution: Number of points to sample in each dimension
        """
        self.weight_ranges = weight_ranges or [[0.1, 3.0], [0.1, 3.0], [0.1, 3.0]]
        self.resolution = resolution
        self.iterations = 0

        logger.info(f"Grid search optimizer initialized with ranges {self.weight_ranges}, resolution {resolution}")

    def optimize_weights(self,
                         training_data: Optional[Dict],
                         validation_queries: Dict[str, str],
                         validation_qrels: Dict[str, Dict[str, int]],
                         evaluation_function: Callable[[Tuple[float, float, float]], float]) -> Tuple[float, float, float]:
        """
        Learn optimal weights using grid search.

        Args:
            training_data: Training dataset (not used)
            validation_queries: Validation queries (not used directly)
            validation_qrels: Validation relevance judgments (not used directly)
            evaluation_function: Function that takes (alpha, beta, gamma) and returns metric score

        Returns:
            Optimal (α, β, γ) weights
        """

        # Generate grid points
        α_values = np.linspace(self.weight_ranges[0][0], self.weight_ranges[0][1], self.resolution)
        β_values = np.linspace(self.weight_ranges[1][0], self.weight_ranges[1][1], self.resolution)
        γ_values = np.linspace(self.weight_ranges[2][0], self.weight_ranges[2][1], self.resolution)

        total_combinations = len(α_values) * len(β_values) * len(γ_values)
        logger.info(f"Grid search over {total_combinations} combinations...")

        best_weights = (1.0, 1.0, 1.0)
        best_score = -np.inf
        evaluated = 0

        try:
            for α in α_values:
                for β in β_values:
                    for γ in γ_values:
                        weights_tuple = (α, β, γ)

                        try:
                            score = evaluation_function(weights_tuple)

                            if score > best_score:
                                best_score = score
                                best_weights = weights_tuple
                                logger.debug(f"New best: weights={weights_tuple}, score={score:.4f}")

                            evaluated += 1
                            if evaluated % 100 == 0:
                                logger.info(f"Evaluated {evaluated}/{total_combinations} combinations")

                        except Exception as e:
                            logger.warning(f"Error evaluating weights {weights_tuple}: {e}")
                            continue

            self.iterations = evaluated

            logger.info("Grid search completed!")
            logger.info(f"Best weights: alpha={best_weights[0]:.3f}, beta={best_weights[1]:.3f}, gamma={best_weights[2]:.3f}")
            logger.info(f"Best score: {best_score:.4f}")
            logger.info(f"Evaluated {evaluated}/{total_combinations} combinations")

            return best_weights

        except Exception as e:
            logger.error(f"Grid search failed: {e}")
            logger.info("Returning baseline weights (1.0, 1.0, 1.0)")
            return (1.0, 1.0, 1.0)


class RandomSearchOptimizer(WeightOptimizer):
    """
    Random search optimizer for learning expansion weights.
    Randomly samples weight combinations within specified bounds.
    """

    def __init__(self,
                 bounds: List[Tuple[float, float]] = None,
                 num_samples: int = 1000,
                 random_seed: int = 42):
        """
        Initialize random search optimizer.

        Args:
            bounds: List of (min, max) bounds for each weight
            num_samples: Number of random samples to evaluate
            random_seed: Random seed for reproducibility
        """
        self.bounds = bounds or [(0.1, 5.0), (0.1, 5.0), (0.1, 5.0)]
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.iterations = 0

        np.random.seed(random_seed)
        logger.info(f"Random search optimizer initialized with {num_samples} samples")

    def optimize_weights(self,
                         training_data: Optional[Dict],
                         validation_queries: Dict[str, str],
                         validation_qrels: Dict[str, Dict[str, int]],
                         evaluation_function: Callable[[Tuple[float, float, float]], float]) -> Tuple[float, float, float]:
        """
        Learn optimal weights using random search.

        Args:
            training_data: Training dataset (not used)
            validation_queries: Validation queries (not used directly)
            validation_qrels: Validation relevance judgments (not used directly)
            evaluation_function: Function that takes (alpha, beta, gamma) and returns metric score

        Returns:
            Optimal (α, β, γ) weights
        """

        logger.info(f"Random search over {self.num_samples} samples...")

        best_weights = (1.0, 1.0, 1.0)
        best_score = -np.inf
        evaluated = 0

        try:
            for i in range(self.num_samples):
                # Sample random weights within bounds
                α = np.random.uniform(self.bounds[0][0], self.bounds[0][1])
                β = np.random.uniform(self.bounds[1][0], self.bounds[1][1])
                γ = np.random.uniform(self.bounds[2][0], self.bounds[2][1])

                weights_tuple = (α, β, γ)

                try:
                    score = evaluation_function(weights_tuple)

                    if score > best_score:
                        best_score = score
                        best_weights = weights_tuple
                        logger.debug(f"New best: weights={weights_tuple}, score={score:.4f}")

                    evaluated += 1
                    if evaluated % 200 == 0:
                        logger.info(f"Evaluated {evaluated}/{self.num_samples} samples")

                except Exception as e:
                    logger.warning(f"Error evaluating weights {weights_tuple}: {e}")
                    continue

            self.iterations = evaluated

            logger.info("Random search completed!")
            logger.info(f"Best weights: alpha={best_weights[0]:.3f}, beta={best_weights[1]:.3f}, gamma={best_weights[2]:.3f}")
            logger.info(f"Best score: {best_score:.4f}")
            logger.info(f"Evaluated {evaluated}/{self.num_samples} samples")

            return best_weights

        except Exception as e:
            logger.error(f"Random search failed: {e}")
            logger.info("Returning baseline weights (1.0, 1.0, 1.0)")
            return (1.0, 1.0, 1.0)


def create_optimizer(optimizer_type: str = 'lbfgs', **kwargs) -> WeightOptimizer:
    """
    Factory function to create weight optimizer.

    Args:
        optimizer_type: Type of optimizer ('lbfgs', 'grid', 'random')
        **kwargs: Additional arguments for the optimizer

    Returns:
        WeightOptimizer instance

    Raises:
        ValueError: If unknown optimizer type
    """
    if optimizer_type.lower() == 'lbfgs':
        return LBFGSOptimizer(**kwargs)
    elif optimizer_type.lower() == 'grid':
        return GridSearchOptimizer(**kwargs)
    elif optimizer_type.lower() == 'random':
        return RandomSearchOptimizer(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for example
    logging.basicConfig(level=logging.INFO)

    print("Weight Optimizer Module Example")
    print("=" * 40)

    # Mock evaluation function for demonstration
    def mock_evaluation_function(weights: Tuple[float, float, float]) -> float:
        """Mock evaluation that peaks around alpha=1.2, beta=0.8, gamma=1.5"""
        alpha, beta, gamma = weights
        target = np.array([1.2, 0.8, 1.5])
        current = np.array([alpha, beta, gamma])
        distance = np.linalg.norm(current - target)
        return 0.7 - 0.1 * distance  # Mock nDCG score

    # Mock data
    mock_training_data = None
    mock_queries = {"q1": "test query"}
    mock_qrels = {"q1": {"doc1": 1}}

    print("Testing different optimizers:")
    print()

    # Test L-BFGS-B optimizer
    print("1. L-BFGS-B Optimizer:")
    lbfgs_optimizer = LBFGSOptimizer(max_iterations=20)
    optimal_weights = lbfgs_optimizer.optimize_weights(
        mock_training_data, mock_queries, mock_qrels, mock_evaluation_function
    )
    print(f"   Optimal weights: alpha={optimal_weights[0]:.3f}, beta={optimal_weights[1]:.3f}, gamma={optimal_weights[2]:.3f}")
    print()

    # Test Grid Search optimizer
    print("2. Grid Search Optimizer:")
    grid_optimizer = GridSearchOptimizer(resolution=5)  # Small grid for demo
    optimal_weights = grid_optimizer.optimize_weights(
        mock_training_data, mock_queries, mock_qrels, mock_evaluation_function
    )
    print(f"   Optimal weights: alpha={optimal_weights[0]:.3f}, beta={optimal_weights[1]:.3f}, gamma={optimal_weights[2]:.3f}")
    print()

    # Test Random Search optimizer
    print("3. Random Search Optimizer:")
    random_optimizer = RandomSearchOptimizer(num_samples=100)  # Small sample for demo
    optimal_weights = random_optimizer.optimize_weights(
        mock_training_data, mock_queries, mock_qrels, mock_evaluation_function
    )
    print(f"   Optimal weights: α={optimal_weights[0]:.3f}, β={optimal_weights[1]:.3f}, γ={optimal_weights[2]:.3f}")
    print()

    # Test factory function
    print("4. Factory Function:")
    optimizer = create_optimizer('lbfgs', max_iterations=10)
    print(f"   Created optimizer: {type(optimizer).__name__}")

    print("\nAll optimizers completed successfully!")