from __future__ import annotations

import time
from datetime import datetime
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# =============================================================================
# INDIVIDUAL
# =============================================================================

class Individual:
    """
    One candidate design.

    Attributes
    ----------
    params : dict[str, float]
        Design parameters -- one float per dimension, within search space bounds.
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
    Wrap a pipeline evaluator into the (params, generation, max_generations) -> (fitness, result)
    signature expected by CMAEvolutionStrategy.

    Assigns a unique sample_id per call so geometry outputs don't overwrite each other.
    Returns (penalty_fitness, result_dict) on pipeline failure rather than raising.
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
# CMA-ES: Covariance Matrix Adaptation Evolution Strategy
# =============================================================================
#
# Replaces the custom (mu+lambda)-ES with CMA-ES, which adapts the full
# covariance matrix of the search distribution rather than independent
# per-parameter step sizes.
#
# Why CMA-ES:
#   - Per-parameter sigma-adaptation cannot learn correlations between params.
#     Node positions are structurally correlated -- shifting one node should
#     co-vary with adjacent nodes. CMA-ES discovers these correlations
#     automatically via covariance matrix adaptation.
#   - The sigma-collapse observed in the (mu+lambda)-ES (0.38 -> 0.07 over
#     200 generations) is a known failure mode of uncorrelated step-size
#     adaptation. CMA-ES does not have this pathology.
#   - Parameters are normalised to [0,1]^n before passing to CMA-ES, giving
#     consistent step-size semantics regardless of individual parameter ranges.
#
# The evaluate_fn interface, top-k tracking, history format, and result dict
# are identical to the old EvolutionStrategy.run(), so all export code is
# unchanged.

import cma as _cma_lib


@dataclass
class CMAESConfig:
    """
    Configuration for CMA-ES optimiser.

    Key parameters:
        popsize       : population size (lambda). CMA-ES default for n params is
                        4 + floor(3*ln(n)); setting ~30 gives 2x extra exploration.
        n_generations : hard generation limit. Budget = popsize x n_generations.
        sigma_init    : initial step size as fraction of [0,1] normalised space.
                        0.25 means CMA-ES starts with sigma=0.25 in normalised coords.
        sigma_min     : stop if sigma drops below this (convergence criterion).
        stagnation_limit: used for the reference line in the analysis plot only;
                        CMA-ES manages its own convergence via tolx/tolfun.
    """
    popsize:          int   = 30
    n_generations:    int   = 250
    sigma_init:       float = 0.25
    sigma_min:        float = 1e-8    # maps to CMA-ES tolx
    tolfun:           float = 1e-11   # stop if fitness spread < this
    stagnation_limit: int   = 30      # for analysis plot reference line only
    top_k_size:       int   = 10
    log_every:        int   = 5
    verbose:          bool  = True
    penalty_fitness:  float = 1e6
    n_restarts_max:   int   = 0       # CMA-ES handles convergence internally

    def __post_init__(self):
        # Compatibility attributes read by run_export in c23_ga_analysis_export.py
        self.mu  = self.popsize // 2
        self.lam = self.popsize


class CMAEvolutionStrategy:
    """
    CMA-ES wrapper with the same run() interface as EvolutionStrategy.

    Parameters are normalised to [0,1]^n internally so CMA-ES step sizes
    are scale-invariant across parameters with different physical ranges.
    The evaluate_fn always receives parameters in the original physical units.

    Result dict is identical to EvolutionStrategy.run() so the notebook's
    analysis and export cells require no changes.
    """

    def __init__(
        self,
        search_space: dict[str, tuple[float, float]],
        evaluate_fn:  Callable[[dict], tuple[float, dict]],
        config:       CMAESConfig = None,
        seed:         int         = None,
    ):
        self.search_space = search_space
        self.evaluate_fn  = evaluate_fn
        self.config       = config or CMAESConfig()
        self.param_names  = list(search_space.keys())
        self.n_params     = len(self.param_names)
        self.bounds_lo    = np.array([search_space[k][0] for k in self.param_names])
        self.bounds_hi    = np.array([search_space[k][1] for k in self.param_names])
        self.seed         = seed

        # Run state
        self.history:    list[dict]       = []
        self.top_k:      list[Individual] = []
        self.best_ever:  Individual | None = None
        self.n_evals:    int               = 0
        self.stagnation: int               = 0

        cfg = self.config
        print(
            f"[CMA-ES] Initialised: popsize(lam)={cfg.popsize}  "
            f"n_generations={cfg.n_generations}  n_params={self.n_params}\n"
            f"         sigma_init={cfg.sigma_init}  "
            f"Expected evaluations: {cfg.popsize * cfg.n_generations}"
        )

    # -------------------------------------------------------------------------
    # Normalisation helpers
    # -------------------------------------------------------------------------

    def _norm(self, x_abs: np.ndarray) -> np.ndarray:
        """Physical space -> [0,1]^n."""
        return (x_abs - self.bounds_lo) / (self.bounds_hi - self.bounds_lo)

    def _denorm(self, x_norm: np.ndarray) -> np.ndarray:
        """[0,1]^n -> physical space, clipped to bounds."""
        return self.bounds_lo + np.clip(x_norm, 0.0, 1.0) * (self.bounds_hi - self.bounds_lo)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """
        Run CMA-ES and return a result dict compatible with EvolutionStrategy.run().

        Returns
        -------
        dict with keys: best_individual, best_fitness, best_params,
                        best_eval_result, top_k, history, n_evals,
                        n_generations, n_restarts
        """
        cfg              = self.config
        penalty          = cfg.penalty_fitness
        t_start          = time.time()
        dt_start         = datetime.now()
        start_time_str   = dt_start.strftime("%Y-%m-%d %H:%M:%S")

        # Starting point: random in [0,1]^n
        rng = np.random.default_rng(self.seed)
        x0  = rng.uniform(0.0, 1.0, self.n_params).tolist()

        cma_opts = {
            'bounds':   [[0.0] * self.n_params, [1.0] * self.n_params],
            'maxiter':  cfg.n_generations,
            'popsize':  cfg.popsize,
            'tolx':     cfg.sigma_min,
            'tolfun':   cfg.tolfun,
            'verbose':  -9,   # suppress all CMA-ES internal output
            'seed':     self.seed if self.seed is not None else 42,
        }

        cma_es   = _cma_lib.CMAEvolutionStrategy(x0, cfg.sigma_init, cma_opts)
        generation = 0

        while not cma_es.stop():
            generation += 1
            candidates_norm = cma_es.ask()   # list[list[float]] in [0,1]^n

            fitnesses   = []
            individuals = []

            total_expected = cfg.popsize * cfg.n_generations
            for pop_idx, x_norm in enumerate(candidates_norm, 1):
                x_abs  = self._denorm(np.array(x_norm))
                params = dict(zip(self.param_names, x_abs))

                try:
                    fitness, result = self.evaluate_fn(
                        params, generation, cfg.n_generations
                    )
                except Exception as exc:
                    warnings.warn(f"[CMA-ES] Evaluation failed: {exc}")
                    fitness = penalty
                    result  = {"status": "PENALIZED", "reason": str(exc)}

                fitnesses.append(fitness)
                # Store current CMA sigma as a 1-element array (compatible with
                # Individual.sigma which the export reads via list(best.sigma))
                individuals.append(Individual(
                    params      = params,
                    sigma       = np.array([cma_es.sigma]),
                    fitness     = fitness,
                    eval_result = result,
                    generation  = generation,
                ))
                self.n_evals += 1
                print(
                    f"\r[CMA-ES] eval {self.n_evals:>5}/{total_expected}"
                    f"  |  gen {generation:>3}/{cfg.n_generations}"
                    f"  |  pop {pop_idx:>2}/{cfg.popsize}\n",
                    end="", flush=True,
                )

            cma_es.tell(candidates_norm, fitnesses)

            # Update top-k and stagnation counter
            self._update_top_k(individuals)
            if self.top_k and (
                self.best_ever is None
                or self.top_k[0].fitness < self.best_ever.fitness
            ):
                self.best_ever  = deepcopy(self.top_k[0])
                self.stagnation = 0
            else:
                self.stagnation += 1

            self._record_history(generation, individuals, cma_es.sigma)

            if cfg.verbose and generation % cfg.log_every == 0:
                self._print_summary(generation, individuals, cma_es.sigma)

        elapsed      = time.time() - t_start
        stop_reasons = list(cma_es.stop().keys())

        if self.best_ever is None:
            raise RuntimeError(
                "[CMA-ES] Run completed with no valid evaluations — "
                "every candidate was penalized. Check the pipeline configuration."
            )

        print(
            f"\n[CMA-ES] Finished in {elapsed / 60:.1f} min  |  "
            f"{self.n_evals} evaluations  |  {generation} generations\n"
            f"[CMA-ES] Stop reason(s): {stop_reasons}\n"
            f"[CMA-ES] Best fitness: {self.best_ever.fitness:.6f}"
        )
        for rank, ind in enumerate(self.top_k, 1):
            gnn = (ind.eval_result or {}).get("gnn_feasibility", float("nan"))
            print(
                f"     #{rank:2d}  fitness={ind.fitness:.4f}  "
                f"GNN={gnn:.3f}  gen={ind.generation}"
            )

        return {
            "best_individual":  self.best_ever,
            "best_fitness":     self.best_ever.fitness,
            "best_params":      self.best_ever.params,
            "best_eval_result": self.best_ever.eval_result,
            "top_k":            self.top_k,
            "history":          self.history,
            "n_evals":          self.n_evals,
            "n_generations":    generation,
            "n_restarts":       0,
            "start_time":       start_time_str,
            "elapsed_seconds":  round(time.time() - t_start, 1),
        }

    # -------------------------------------------------------------------------
    # Top-k tracking (identical logic to EvolutionStrategy)
    # -------------------------------------------------------------------------

    @staticmethod
    def _top_k_key(ind: Individual) -> tuple:
        return tuple(round(v, 8) for v in ind.params.values())

    def _update_top_k(self, candidates: list[Individual]) -> None:
        k    = self.config.top_k_size
        seen = {self._top_k_key(ind) for ind in self.top_k}
        for ind in candidates:
            if ind.fitness is None or ind.fitness >= self.config.penalty_fitness:
                continue
            key = self._top_k_key(ind)
            if key not in seen:
                self.top_k.append(deepcopy(ind))
                seen.add(key)
        self.top_k.sort(key=lambda x: x.fitness)
        self.top_k = self.top_k[:k]

    # -------------------------------------------------------------------------
    # History logging (same schema as EvolutionStrategy._record_history)
    # -------------------------------------------------------------------------

    def _record_history(
        self, gen: int, individuals: list[Individual], sigma: float
    ) -> None:
        penalty   = self.config.penalty_fitness
        fitnesses = [ind.fitness for ind in individuals]
        valid     = [ind for ind in individuals if ind.fitness < penalty]

        best_gen = min(valid, key=lambda x: x.fitness) if valid else None
        bfr = ((best_gen.eval_result or {}).get("fitness_result") or {}) if best_gen else {}
        ber = (best_gen.eval_result or {}) if best_gen else {}

        def _pop_mean(key):
            vals = [
                float((ind.eval_result or {}).get(key, float("nan")))
                for ind in valid
                if (ind.eval_result or {}).get(key) is not None
            ]
            return float(np.mean(vals)) if vals else float("nan")

        self.history.append({
            "generation":             gen,
            "n_evals":                self.n_evals,
            "best_fitness":           float(np.min(fitnesses)),
            "mean_fitness":           float(np.mean(fitnesses)),
            "worst_fitness":          float(np.max(fitnesses)),
            "std_fitness":            float(np.std(fitnesses)),
            "mean_sigma":             float(sigma),
            "stagnation":             self.stagnation,
            "best_ever":              self.best_ever.fitness if self.best_ever else None,
            "w_structural":           float(ber.get("w_structural",   float("nan"))),
            "n_penalty":              sum(1 for ind in individuals if ind.fitness >= penalty),
            "best_cost_norm":         float(bfr.get("cost_norm",                float("nan"))),
            "best_reuse_norm":        float(bfr.get("reuse_norm",               float("nan"))),
            "best_structural_infeas": float(bfr.get("structural_infeasibility", float("nan"))),
            "mean_gnn":               _pop_mean("gnn_feasibility"),
            "mean_cost":              _pop_mean("total_cost"),
            "mean_reuse":             _pop_mean("reuse_fraction"),
        })

    def _print_summary(
        self, gen: int, individuals: list[Individual], sigma: float
    ) -> None:
        fitnesses   = [ind.fitness for ind in individuals]
        best_ever_f = self.best_ever.fitness if self.best_ever else float("nan")
        print(
            f"\n[CMA-ES] Gen {gen:3d}  "
            f"best={min(fitnesses):.4f}  mean={float(np.mean(fitnesses)):.4f}  "
            f"best_ever={best_ever_f:.4f}  "
            f"sigma={sigma:.5f}  stag={self.stagnation}  evals={self.n_evals}"
        )
