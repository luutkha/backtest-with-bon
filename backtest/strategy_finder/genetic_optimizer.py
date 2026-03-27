"""
Genetic Algorithm Optimizer for Strategy Parameters.

Uses a genetic algorithm to efficiently search parameter space
instead of exhaustive grid search.
"""

import random
import logging
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Gene:
    """Represents a single gene (parameter) in a chromosome"""
    name: str
    value: Any
    min_val: Any
    max_val: Any
    is_discrete: bool = True  # For int/float params


@dataclass
class Chromosome:
    """Represents a complete parameter set (individual in population)"""
    genes: Dict[str, Gene]
    fitness: float = 0.0

    def to_params(self) -> Dict[str, Any]:
        """Convert chromosome to parameter dict"""
        return {name: gene.value for name, gene in self.genes.items()}

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for crossover operations"""
        values = []
        for gene in self.genes.values():
            if gene.is_discrete:
                values.append(float(gene.value))
            else:
                values.append(float(gene.value))
        return np.array(values)


class GeneticOptimizer:
    """
    Genetic algorithm optimizer for strategy parameters.

    Workflow:
    1. Initialize population with random parameter sets
    2. Evaluate fitness (backtest Sharpe ratio)
    3. Select top performers
    4. Crossover to create offspring
    5. Mutate some genes
    6. Repeat for N generations
    """

    def __init__(
        self,
        param_ranges: Dict[str, List[Any]],
        fitness_func: Callable[[Dict[str, Any]], float],
        population_size: int = 50,
        generations: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elite_ratio: float = 0.2,
        tournament_size: int = 3,
        verbose: bool = True,
    ):
        """
        Initialize GA optimizer.

        Args:
            param_ranges: Dict of param name -> list of possible values
            fitness_func: Function that takes params and returns fitness score
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover
            elite_ratio: Ratio of top performers to keep unchanged
            tournament_size: Number of individuals in tournament selection
            verbose: Enable logging
        """
        self.param_ranges = param_ranges
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_ratio = elite_ratio
        self.tournament_size = tournament_size
        self.verbose = verbose

        self.population: List[Chromosome] = []
        self.best_chromosome: Optional[Chromosome] = None
        self.history: List[float] = []  # Best fitness per generation

    def _create_gene(self, name: str, values: List[Any]) -> Tuple[Gene, List[Any]]:
        """Create a gene with range information"""
        # Determine if discrete or continuous
        is_discrete = all(isinstance(v, (int, str)) for v in values)

        if is_discrete:
            min_val = min(values) if isinstance(values[0], (int, float)) else 0
            max_val = max(values) if isinstance(values[0], (int, float)) else len(values) - 1
        else:
            min_val = min(values)
            max_val = max(values)

        return Gene(
            name=name,
            value=random.choice(values),
            min_val=min_val,
            max_val=max_val,
            is_discrete=is_discrete
        ), values

    def _initialize_population(self) -> List[Chromosome]:
        """Initialize random population"""
        population = []
        param_values_list = {}

        for name, values in self.param_ranges.items():
            param_values_list[name] = values

        for _ in range(self.population_size):
            genes = {}
            for name, values in param_values_list.items():
                gene, _ = self._create_gene(name, values)
                genes[name] = gene
            population.append(Chromosome(genes=genes))

        return population

    def _evaluate_population(self) -> None:
        """Evaluate fitness for all individuals in population"""
        for chromosome in self.population:
            if chromosome.fitness == 0:  # Not yet evaluated
                try:
                    params = chromosome.to_params()
                    chromosome.fitness = self.fitness_func(params)
                except Exception as e:
                    logger.warning(f"Fitness evaluation failed: {e}")
                    chromosome.fitness = -999999  # Penalize failure

        # Sort by fitness
        self.population.sort(key=lambda c: c.fitness, reverse=True)

        # Update best
        if self.population and self.population[0].fitness > (self.best_chromosome.fitness if self.best_chromosome else -999999):
            self.best_chromosome = Chromosome(
                genes={n: Gene(n, g.value, g.min_val, g.max_val, g.is_discrete)
                       for n, g in self.population[0].genes.items()},
                fitness=self.population[0].fitness
            )

    def _tournament_selection(self) -> Chromosome:
        """Select individual using tournament selection"""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(tournament, key=lambda c: c.fitness)

    def _crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Blend two parents to create offspring"""
        if random.random() > self.crossover_rate:
            return parent1, parent2

        child1_genes = {}
        child2_genes = {}

        for name in parent1.genes:
            g1 = parent1.genes[name]
            g2 = parent2.genes[name]

            if g1.is_discrete:
                # Discrete crossover: randomly pick from parents
                chosen1 = random.choice([g1.value, g2.value])
                chosen2 = random.choice([g1.value, g2.value])
            else:
                # Blend for continuous values
                val1 = (g1.value + g2.value) / 2
                val2 = (g1.value + g2.value) / 2

                # Add some variation
                val1 += random.uniform(-0.1, 0.1) * (g1.max_val - g1.min_val)
                val2 += random.uniform(-0.1, 0.1) * (g2.max_val - g2.min_val)

                # Clip to valid range
                val1 = max(g1.min_val, min(g1.max_val, val1))
                val2 = max(g2.min_val, min(g2.max_val, val2))

                chosen1, chosen2 = val1, val2

            child1_genes[name] = Gene(name, chosen1, g1.min_val, g1.max_val, g1.is_discrete)
            child2_genes[name] = Gene(name, chosen2, g2.min_val, g2.max_val, g2.is_discrete)

        return Chromosome(genes=child1_genes), Chromosome(genes=child2_genes)

    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        """Mutate chromosome genes"""
        for name, gene in chromosome.genes.items():
            if random.random() < self.mutation_rate:
                values = self.param_ranges[name]
                if gene.is_discrete:
                    chromosome.genes[name].value = random.choice(values)
                else:
                    # Gaussian mutation
                    std = 0.1 * (gene.max_val - gene.min_val)
                    new_val = gene.value + random.gauss(0, std)
                    chromosome.genes[name].value = max(gene.min_val, min(gene.max_val, new_val))

        return chromosome

    def _evolve_generation(self) -> None:
        """Create next generation"""
        elite_count = max(1, int(self.population_size * self.elite_ratio))

        # Keep elite individuals
        next_population = self.population[:elite_count]

        # Create offspring
        while len(next_population) < self.population_size:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            child1, child2 = self._crossover(parent1, parent2)
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)

            next_population.append(child1)
            if len(next_population) < self.population_size:
                next_population.append(child2)

        self.population = next_population

    def run(self) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization.

        Returns:
            Dict with best parameters and fitness history
        """
        if self.verbose:
            logger.info(f"Starting GA: pop={self.population_size}, gen={self.generations}")

        # Initialize
        self.population = self._initialize_population()
        self.best_chromosome = None
        self.history = []

        for gen in range(self.generations):
            # Evaluate
            self._evaluate_population()
            best_fitness = self.population[0].fitness
            self.history.append(best_fitness)

            if self.verbose:
                logger.info(f"Gen {gen+1}/{self.generations}: best_fitness={best_fitness:.4f}")

            # Evolve
            if gen < self.generations - 1:
                self._evolve_generation()

        # Final evaluation
        self._evaluate_population()

        if self.verbose:
            logger.info(f"GA complete. Best fitness: {self.best_chromosome.fitness:.4f}")
            logger.info(f"Best params: {self.best_chromosome.to_params()}")

        return {
            'best_params': self.best_chromosome.to_params() if self.best_chromosome else {},
            'best_fitness': self.best_chromosome.fitness if self.best_chromosome else None,
            'history': self.history,
        }

    def get_best_params(self) -> Dict[str, Any]:
        """Run GA and return best parameters"""
        result = self.run()
        return result['best_params']


def grid_search_with_pruning(
    param_grid: Dict[str, List[Any]],
    fitness_func: Callable[[Dict[str, Any]], float],
    min_trades: int = 10,
    prune_threshold: float = -2.0,
    max_combinations: Optional[int] = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Grid search with early pruning based on partial evaluation.

    For complex parameter spaces, can prune branches that are
    unlikely to yield good results.

    Args:
        param_grid: Dict of param name -> list of values
        fitness_func: Function to evaluate parameters
        min_trades: Minimum trades required for valid result
        prune_threshold: Skip combos with fitness below this
        max_combinations: Limit total combinations (None = all)
        verbose: Enable logging

    Returns:
        List of (params, fitness) tuples, sorted by fitness
    """
    from itertools import product

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    if max_combinations and len(combinations) > max_combinations:
        # Sample if too many combinations
        combinations = random.sample(combinations, max_combinations)

    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))

        if verbose and (i + 1) % 100 == 0:
            logger.info(f"Grid search: {i+1}/{len(combinations)}")

        try:
            fitness = fitness_func(params)
            results.append((params, fitness))
        except Exception as e:
            if verbose:
                logger.warning(f"Failed for {params}: {e}")
            results.append((params, prune_threshold - 1))

    # Sort by fitness
    results.sort(key=lambda x: x[1], reverse=True)

    return results
