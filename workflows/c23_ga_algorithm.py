# =============================================================================
# c30_ga_algorithm_v1.py — (μ+λ) Evolution Strategy for Truss Geometry
# =============================================================================
#
# Optimises continuous geometry parameters (node positions / spans) using a
# (μ+λ) Evolution Strategy with adaptive step sizes (self-adaptive σ).
#
# Why (μ+λ) ES over standard GA:
#   - Expensive evaluations (5–30s each): μ=10, λ=20 means only 20 evaluations
#     per generation — parents are never re-evaluated (elitism is free).
#   - Continuous search space: Gaussian mutation is theoretically grounded for
#     real-valued parameters. Crossover helps escape local optima.
#   - Self-adaptive σ: each individual carries its own step size vector that
#     evolves alongside the design parameters — no manual σ tuning required.
#   - (μ+λ) selection (parents compete with offspring) gives stronger elitism
#     than (μ,λ) (offspring-only), which is appropriate when evaluations are
#     expensive and you can't afford to lose good solutions.
#
# Operator summary:
#   Selection  : tournament selection from (parents ∪ offspring) pool
#   Crossover  : intermediate recombination (arithmetic mean of two parents)
#   Mutation   : Gaussian perturbation with self-adaptive σ per parameter
#   σ update   : log-normal update rule (standard ES self-adaptation)
#
# Usage:
#   from c30_ga_algorithm_v1 import EvolutionStrategy, ESConfig
#
#   cfg = ESConfig(mu=10, lam=20, n_generations=50)
#   es  = EvolutionStrategy(
#       search_space   = optimizer_search_space,
#       evaluate_fn    = evaluate_design_candidate_wrapped,
#       config         = cfg,
#   )
#   result = es.run()
#
# Requires from notebook:
#   evaluate_design_candidate()  — the full pipeline evaluator
#   optimizer_search_space       — dict of {param_name: (lower, upper)}
#   SURROGATE_BUNDLE, df_input_stock, FIXED_NORMALIZATION_CONSTANTS, GA_CONFIG

from __future__ import annotations

import time
import random
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# =============================================================================
# CONFIGURATION DATACLASS
# =============================================================================

@dataclass
class ESConfig:
    """
    All hyperparameters for the (μ+λ) Evolution Strategy.

    Sensible defaults for expensive evaluations (5–30s per call):
        mu=15, lam=30 → 30 evaluations per generation
        n_generations=50 → ~1500 total evaluations → ~9 hours at 20s/eval
        For faster iteration during development: mu=5, lam=10, n_generations=20
    """
    # Population
    mu:            int   = 15      # number of parents kept each generation
    lam:           int   = 30      # number of offspring generated each generation

    # Termination
    n_generations: int   = 50      # maximum generations
    budget:        int   = None    # optional: hard cap on total evaluations

    # Crossover
    crossover_prob: float = 0.7    # probability of applying recombination
                                   # (vs cloning a single parent)

    # Self-adaptive mutation (log-normal σ update)
    sigma_init:    float = 0.15     # initial step size as fraction of param range
    sigma_min:     float = 1e-4    # floor — prevents σ collapsing to zero
    tau:           float = None    # global learning rate (None = 1/sqrt(n_params))
    tau_prime:     float = None    # local  learning rate (None = 1/sqrt(2*n_params))

    # Tournament selection
    tournament_size: int = 3       # k in k-tournament; higher = more selective

    # Restart
    stagnation_limit: int = 15     # restart if best fitness unchanged for this
                                   # many generations (None to disable)
    n_restarts_max:   int = 3      # max restarts before giving up

    # Penalty
    penalty_fitness: float = 1e6   # fitness assigned to infeasible / failed evals

    # Top-k tracking
    top_k_size:    int   = 10      # number of best unique designs retained across all generations

    # Logging
    log_every:     int   = 1       # print summary every N generations
    verbose:       bool  = True


# =============================================================================
# INDIVIDUAL
# =============================================================================

class Individual:
    """
    One candidate design.

    Attributes
    ----------
    params : dict[str, float]
        Design parameters — one float per dimension, within search space bounds.
    sigma  : np.ndarray [n_params]
        Self-adaptive step sizes, one per parameter. Evolved alongside params.
    fitness : float
        Objective value (lower = better). None until evaluated.
    eval_result : dict
        Full output from evaluate_design_candidate() for post-processing.
    """

    __slots__ = ("params", "sigma", "fitness", "eval_result", "generation")

    def __init__(
        self,
        params:      dict[str, float],
        sigma:       np.ndarray,
        fitness:     float = None,
        eval_result: dict  = None,
        generation:  int   = 0,
    ):
        self.params      = params
        self.sigma       = sigma
        self.fitness     = fitness
        self.eval_result = eval_result
        self.generation  = generation

    def is_evaluated(self) -> bool:
        return self.fitness is not None

    def __repr__(self) -> str:
        f = f"{self.fitness:.4f}" if self.fitness is not None else "?"
        return f"Individual(fitness={f}, gen={self.generation})"


# =============================================================================
# EVOLUTION STRATEGY
# =============================================================================

class EvolutionStrategy:
    """
    (μ+λ) Evolution Strategy with self-adaptive Gaussian mutation.

    Parameters
    ----------
    search_space  : dict[str, tuple[float, float]]
        {param_name: (lower_bound, upper_bound)} for each design variable.
    evaluate_fn   : Callable[[dict], float]
        Function that takes design_params dict and returns scalar fitness.
        Lower = better. Should return config.penalty_fitness on failure.
        Wrap evaluate_design_candidate() with _make_evaluate_fn() below.
    config        : ESConfig
    seed          : int | None
    """

    def __init__(
        self,
        search_space: dict[str, tuple[float, float]],
        evaluate_fn:  Callable[[dict], float],
        config:       ESConfig = None,
        seed:         int      = None,
    ):
        self.search_space = search_space
        self.evaluate_fn  = evaluate_fn
        self.config       = config or ESConfig()
        self.param_names  = list(search_space.keys())
        self.n_params     = len(self.param_names)
        self.bounds_lo    = np.array([search_space[k][0] for k in self.param_names])
        self.bounds_hi    = np.array([search_space[k][1] for k in self.param_names])
        self.param_range  = self.bounds_hi - self.bounds_lo

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Learning rates for self-adaptive σ (ES theory defaults)
        cfg = self.config
        n   = self.n_params
        self.tau       = cfg.tau       or 1.0 / np.sqrt(n)
        self.tau_prime = cfg.tau_prime or 1.0 / np.sqrt(2 * n)

        # State
        self.population:    list[Individual] = []
        self.history:       list[dict]       = []   # one entry per generation
        self.best_ever:     Individual | None = None
        self.top_k:         list[Individual] = []   # best k unique designs seen so far
        self.generation:    int               = 0
        self.n_evals:       int               = 0
        self.n_restarts:    int               = 0
        self.stagnation:    int               = 0

        print(
            f"[ES] Initialised: μ={cfg.mu} λ={cfg.lam} "
            f"n_generations={cfg.n_generations} n_params={n}\n"
            f"     τ={self.tau:.4f}  τ'={self.tau_prime:.4f}  "
            f"σ_init={cfg.sigma_init:.4f}  σ_min={cfg.sigma_min:.6f}\n"
            f"     Expected evaluations: "
            f"{cfg.mu + cfg.lam * cfg.n_generations} "
            f"(initial {cfg.mu} + {cfg.lam}×{cfg.n_generations} offspring)"
        )

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """
        Run the full evolution strategy.

        Returns
        -------
        result : dict
            best_individual  — Individual with lowest fitness found
            best_fitness     — float
            best_params      — dict[str, float]
            best_eval_result — full pipeline output dict
            history          — list of per-generation summaries
            n_evals          — total evaluations performed
            n_generations    — generations completed
            n_restarts       — number of restarts triggered
        """
        cfg        = self.config
        t_start    = time.time()
        budget     = cfg.budget or int(1e9)

        # Initialise population
        self.population = self._initialise_population(cfg.mu)
        self._evaluate_population(self.population, generation=0)
        self._update_top_k(self.population)

        if cfg.verbose:
            print(f"\n[ES] Initial population evaluated ({cfg.mu} individuals)")
            self._print_generation_summary(0)

        for gen in range(1, cfg.n_generations + 1):
            if self.n_evals >= budget:
                print(f"[ES] Budget of {budget} evaluations reached at generation {gen}.")
                break

            self.generation = gen

            # Generate offspring
            offspring = self._generate_offspring(cfg.lam)

            # Evaluate offspring (parents are NOT re-evaluated — free elitism)
            self._evaluate_population(offspring, generation=gen)

            # (μ+λ) selection: best μ from parents ∪ offspring
            combined = self.population + offspring
            self._update_top_k(combined)   # capture all evaluated before selection discards any
            self.population = self._select(combined, cfg.mu)

            # Track best ever
            gen_best = min(self.population, key=lambda ind: ind.fitness)
            if self.best_ever is None or gen_best.fitness < self.best_ever.fitness:
                self.best_ever  = deepcopy(gen_best)
                self.stagnation = 0
            else:
                self.stagnation += 1

            # Log
            self._record_history(gen)
            if cfg.verbose and gen % cfg.log_every == 0:
                self._print_generation_summary(gen)

            # Stagnation restart
            if (cfg.stagnation_limit is not None
                    and self.stagnation >= cfg.stagnation_limit
                    and self.n_restarts < cfg.n_restarts_max):
                self._restart()

        elapsed = time.time() - t_start
        print(
            f"\n[ES] Finished in {elapsed/60:.1f} min  |  "
            f"{self.n_evals} evaluations  |  "
            f"{self.generation} generations  |  "
            f"{self.n_restarts} restarts\n"
            f"[ES] Best fitness: {self.best_ever.fitness:.6f}"
        )
        print(f"[ES] Top-{len(self.top_k)} designs:")
        for rank, ind in enumerate(self.top_k, 1):
            gnn = (ind.eval_result or {}).get("gnn_feasibility", float("nan"))
            print(f"     #{rank:2d}  fitness={ind.fitness:.4f}  GNN={gnn:.3f}  gen={ind.generation}")

        return {
            "best_individual":  self.best_ever,
            "best_fitness":     self.best_ever.fitness,
            "best_params":      self.best_ever.params,
            "best_eval_result": self.best_ever.eval_result,
            "top_k":            self.top_k,
            "history":          self.history,
            "n_evals":          self.n_evals,
            "n_generations":    self.generation,
            "n_restarts":       self.n_restarts,
        }

    # -------------------------------------------------------------------------
    # INITIALISATION
    # -------------------------------------------------------------------------

    def _initialise_population(self, n: int) -> list[Individual]:
        """Sample n individuals uniformly at random within bounds."""
        population = []
        sigma_init = self.config.sigma_init * self.param_range

        for _ in range(n):
            params_arr = self.bounds_lo + np.random.uniform(0, 1, self.n_params) * self.param_range
            params     = dict(zip(self.param_names, params_arr))
            sigma      = sigma_init.copy()
            population.append(Individual(params=params, sigma=sigma, generation=0))

        return population

    # -------------------------------------------------------------------------
    # OPERATORS
    # -------------------------------------------------------------------------

    def _recombine(self, p1: Individual, p2: Individual) -> tuple[np.ndarray, np.ndarray]:
        """
        Intermediate recombination: offspring inherits arithmetic mean of
        parents for both params and sigma.

        Why intermediate (not discrete) recombination:
        - Continuous parameters → the mean of two valid parents is also valid
          within bounds (after clipping).
        - Averages σ vectors too, giving a conservative initial step size that
          is then refined by mutation.
        """
        arr1 = np.array([p1.params[k] for k in self.param_names])
        arr2 = np.array([p2.params[k] for k in self.param_names])
        child_arr   = 0.5 * (arr1 + arr2)
        child_sigma = 0.5 * (p1.sigma + p2.sigma)
        return child_arr, child_sigma

    def _mutate(self, params_arr: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Self-adaptive Gaussian mutation (ES standard).

        σ update (log-normal rule):
            σ_i' = σ_i * exp(τ' * N(0,1) + τ * N_i(0,1))
            where τ' is the global learning rate, τ is the local learning rate,
            N(0,1) is shared across all σ_i, N_i(0,1) is individual per σ_i.

        Parameter update:
            x_i' = x_i + σ_i' * N_i(0,1)

        The log-normal update ensures σ stays positive and can both increase
        and decrease — allowing the step size to self-tune during evolution.
        """
        # Update σ first (before applying to params — ES convention)
        global_noise = np.random.randn()
        local_noise  = np.random.randn(self.n_params)
        new_sigma    = sigma * np.exp(
            self.tau_prime * global_noise + self.tau * local_noise
        )
        new_sigma    = np.maximum(new_sigma, self.config.sigma_min)

        # Perturb parameters
        new_params_arr = params_arr + new_sigma * np.random.randn(self.n_params)

        # Clip to bounds (reflection would be theoretically cleaner but
        # clipping is simpler and works well in practice for box constraints)
        new_params_arr = np.clip(new_params_arr, self.bounds_lo, self.bounds_hi)

        return new_params_arr, new_sigma

    def _generate_offspring(self, n: int) -> list[Individual]:
        """Generate n offspring via recombination + mutation."""
        cfg      = self.config
        offspring = []

        for _ in range(n):
            # Select parent(s)
            if len(self.population) >= 2 and np.random.rand() < cfg.crossover_prob:
                p1, p2       = random.sample(self.population, 2)
                params_arr, sigma = self._recombine(p1, p2)
            else:
                p1           = random.choice(self.population)
                params_arr   = np.array([p1.params[k] for k in self.param_names])
                sigma        = p1.sigma.copy()

            # Mutate
            params_arr, sigma = self._mutate(params_arr, sigma)
            params            = dict(zip(self.param_names, params_arr))

            offspring.append(Individual(
                params     = params,
                sigma      = sigma,
                generation = self.generation,
            ))

        return offspring

    # -------------------------------------------------------------------------
    # SELECTION
    # -------------------------------------------------------------------------

    def _select(self, pool: list[Individual], n: int) -> list[Individual]:
        """
        Tournament selection: run n tournaments of size k, each returning
        the winner (lowest fitness). Used for (μ+λ) survivor selection.

        Tournament selection is preferred over truncation (top-n) here because:
        - It maintains more diversity — worse individuals occasionally survive.
        - It avoids premature convergence when several individuals have very
          similar fitness values near the optimum.
        """
        k        = min(self.config.tournament_size, len(pool))
        selected = []
        for _ in range(n):
            tournament = random.sample(pool, k)
            winner     = min(tournament, key=lambda ind: ind.fitness)
            selected.append(deepcopy(winner))
        return selected

    # -------------------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------------------

    def _evaluate_population(self, population: list[Individual], generation: int = 0) -> None:
        """Evaluate each unevaluated individual in the population in-place."""
        to_eval = [ind for ind in population if not ind.is_evaluated()]
        n       = len(to_eval)

        if n == 0:
            return

        for i, ind in enumerate(to_eval):
            t0 = time.time()
            try:
                fitness, eval_result = self.evaluate_fn(ind.params, generation, self.config.n_generations)
            except Exception as exc:
                warnings.warn(f"[ES] Evaluation failed: {exc}")
                fitness     = self.config.penalty_fitness
                eval_result = {"status": "PENALIZED", "reason": str(exc)}

            ind.fitness     = fitness
            ind.eval_result = eval_result
            self.n_evals   += 1
            elapsed         = time.time() - t0

            if self.config.verbose:
                status = eval_result.get("status", "?") if eval_result else "?"
                print(
                    f"  [{self.n_evals:4d}] gen={self.generation} "
                    f"ind={i+1}/{n}  fitness={fitness:.4f}  "
                    f"status={status}  ({elapsed:.1f}s)"
                )

    # -------------------------------------------------------------------------
    # TOP-K TRACKING
    # -------------------------------------------------------------------------

    @staticmethod
    def _top_k_key(ind: Individual) -> tuple:
        """Stable identity key based on rounded param values — used for dedup."""
        return tuple(round(v, 8) for v in ind.params.values())

    def _update_top_k(self, candidates: list[Individual]) -> None:
        """
        Merge evaluated candidates into self.top_k, keeping the best k unique
        designs by fitness. Deduplication is by parameter identity so the same
        design doesn't occupy multiple slots (e.g. elites that survive multiple
        generations).
        """
        k = self.config.top_k_size
        seen_keys = {self._top_k_key(ind) for ind in self.top_k}
        for ind in candidates:
            if ind.fitness is None:
                continue
            key = self._top_k_key(ind)
            if key not in seen_keys:
                self.top_k.append(deepcopy(ind))
                seen_keys.add(key)
        self.top_k.sort(key=lambda x: x.fitness)
        self.top_k = self.top_k[:k]

    # -------------------------------------------------------------------------
    # RESTART
    # -------------------------------------------------------------------------

    def _restart(self) -> None:
        """
        Partial restart: keep the best μ//2 individuals, reseed the rest.
        Resets σ for reseeded individuals to initial values.
        Preserves the best solution found so far.
        """
        cfg          = self.config
        n_keep       = cfg.mu // 2
        sorted_pop   = sorted(self.population, key=lambda ind: ind.fitness)
        keep         = sorted_pop[:n_keep]
        new_randoms  = self._initialise_population(cfg.mu - n_keep)

        self.population  = keep + new_randoms
        self.stagnation  = 0
        self.n_restarts += 1

        print(
            f"\n[ES] Restart {self.n_restarts}/{cfg.n_restarts_max} "
            f"at generation {self.generation}  "
            f"(kept {n_keep} elites, reseeded {cfg.mu - n_keep})"
        )

        # Evaluate new random individuals
        self._evaluate_population(new_randoms, generation=self.generation)

    # -------------------------------------------------------------------------
    # LOGGING
    # -------------------------------------------------------------------------

    def _record_history(self, gen: int) -> None:
        fitnesses = [ind.fitness for ind in self.population]
        sigmas    = np.array([ind.sigma for ind in self.population])
        entry     = {
            "generation":    gen,
            "n_evals":       self.n_evals,
            "best_fitness":  float(np.min(fitnesses)),
            "mean_fitness":  float(np.mean(fitnesses)),
            "worst_fitness": float(np.max(fitnesses)),
            "std_fitness":   float(np.std(fitnesses)),
            "mean_sigma":    float(sigmas.mean()),
            "stagnation":    self.stagnation,
            "best_ever":     self.best_ever.fitness if self.best_ever else None,
        }
        self.history.append(entry)

    def _print_generation_summary(self, gen: int) -> None:
        fitnesses   = [ind.fitness for ind in self.population]
        best_f      = min(fitnesses)
        mean_f      = np.mean(fitnesses)
        best_ever_f = self.best_ever.fitness if self.best_ever else float("nan")
        mean_sigma  = np.mean([ind.sigma for ind in self.population])

        print(
            f"[ES] Gen {gen:3d}  "
            f"best={best_f:.4f}  mean={mean_f:.4f}  "
            f"best_ever={best_ever_f:.4f}  "
            f"σ_mean={mean_sigma:.4f}  "
            f"stag={self.stagnation}  "
            f"evals={self.n_evals}"
        )


# =============================================================================
# EVALUATE FUNCTION WRAPPER
# =============================================================================

def make_evaluate_fn(
    evaluate_fn_raw:       Callable,
    df_stock:              Any,
    fixed_norm_constants:  dict,
    config_dict:           dict,
    bundle:                Any  = None,
    sample_id_offset:      int  = 0,
    prepared_gnn_stock:    Any  = None,
    verbose:               bool = False,
) -> Callable[[dict], tuple[float, dict]]:
    """
    Wrap a pipeline evaluator into the (params → fitness, result) signature
    expected by EvolutionStrategy.

    Parameters
    ----------
    evaluate_fn_raw : the evaluate_design_candidate function from the notebook.
        Passed explicitly so this module has no dependency on notebook globals.

    The wrapper:
    - Assigns a unique sample_id per call (offset + call counter) so geometry
      outputs don't overwrite each other if export_dir is set.
    - Returns (penalty_fitness, result_dict) on pipeline failure rather than
      raising — the ES handles failed evaluations gracefully.

    Usage:
        evaluate_fn = make_evaluate_fn(
            evaluate_fn_raw      = evaluate_design_candidate,
            df_stock             = df_input_stock,
            bundle               = SURROGATE_BUNDLE,
            fixed_norm_constants = FIXED_NORMALIZATION_CONSTANTS,
            config_dict          = GA_CONFIG,
        )
        es = EvolutionStrategy(
            search_space = es_search_space,
            evaluate_fn  = evaluate_fn,
            config       = ESConfig(mu=10, lam=20, n_generations=50),
        )
    """
    call_counter = [0]
    penalty      = float(config_dict.get("penalty_fitness", 1e6))

    def _evaluate(design_params: dict, generation: int = 0, max_generations: int = 1) -> tuple[float, dict]:
        call_counter[0] += 1
        sid = sample_id_offset + call_counter[0]

        result = evaluate_fn_raw(
            design_params        = design_params,
            df_stock             = df_stock,
            bundle               = bundle,
            fixed_norm_constants = fixed_norm_constants,
            config_dict          = config_dict,
            sample_id            = sid,
            verbose              = verbose,
            prepared_gnn_stock   = prepared_gnn_stock,
            generation           = generation,
            max_generations      = max_generations,
        )

        fitness = float(result.get("fitness", penalty))
        return fitness, result

    return _evaluate


# =============================================================================
# CONVERGENCE PLOT HELPER
# =============================================================================

def plot_convergence(history: list[dict], config: ESConfig = None) -> None:
    """
    Plot best / mean / worst fitness and mean σ over generations.
    Call after es.run() with the returned history list.
    """
    import matplotlib.pyplot as plt
    try:
        from config import PLOT_COLORS as C, PLOT_STYLE as S
    except ImportError:
        C = {"primary": "#61788C", "accent": "#F2994B",
             "danger": "#D9653B", "secondary": "#9CA5A6",
             "neutral": "#D7D9D9", "black": "#000000"}
        S = {"figsize_medium": (12, 7), "dpi": 100, "grid_alpha": 0.3,
             "line_width": 2.0}

    gens      = [h["generation"]    for h in history]
    best      = [h["best_fitness"]  for h in history]
    mean      = [h["mean_fitness"]  for h in history]
    best_ever = [h["best_ever"]     for h in history]
    sigmas    = [h["mean_sigma"]    for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=S["figsize_medium"])
    fig.suptitle("Evolution Strategy — Convergence", fontweight="bold", fontsize=13)

    ax1.plot(gens, best_ever, color=C["primary"],   lw=S["line_width"],
             label="Best ever")
    ax1.plot(gens, best,      color=C["accent"],    lw=S["line_width"],
             linestyle="--", label="Gen best")
    ax1.plot(gens, mean,      color=C["secondary"], lw=S["line_width"],
             linestyle=":",  label="Gen mean")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness (lower = better)")
    ax1.set_title("Fitness Convergence")
    ax1.legend()
    ax1.grid(True, alpha=S["grid_alpha"])
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(gens, sigmas, color=C["primary"], lw=S["line_width"])
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Mean σ (step size)")
    ax2.set_title("Self-Adaptive Step Size")
    ax2.grid(True, alpha=S["grid_alpha"])
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig