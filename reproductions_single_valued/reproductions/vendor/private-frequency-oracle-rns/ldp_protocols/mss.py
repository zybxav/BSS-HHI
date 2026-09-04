from __future__ import annotations

# Standard imports
from typing import Iterable, List, Optional, Tuple
from joblib import Parallel, delayed
import multiprocessing

# Third-party imports
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix, vstack
import scipy.sparse as sp
from scipy.sparse.linalg import lsmr, svds
from scipy.optimize import lsq_linear
from numba import jit, int64
from numba.typed import List

# Our imports
from ldp_protocols.base import BaseProtocol
from ldp_protocols.ss import SubsetSelection, ss_obfuscate
from utils.cache import load_mod_cache, save_mod_cache, get_mod_cache_key

# Load the moduli cache if it exists
_MOD_CACHE = load_mod_cache()

# Some helpers
def _safe_sparse_pinv(A, lam: float = 1e-6) -> np.ndarray:
    """
    Robust computation of the pseudo-inverse of a matrix A.

    Falls back to several alternative methods if SVD fails, and supports
    sparse matrix inputs.

    Parameters
    ----------
    A : array-like or sparse matrix
        Input matrix to pseudo-invert.
    lam : float
        Regularization parameter to improve numerical stability.

    Returns
    -------
    np.ndarray
        Pseudo-inverse of matrix A.
    """

    if sp.issparse(A):
        A = A.toarray()
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape
    I_m = np.eye(m, dtype=np.float64)
    try:
        U, s, VT = np.linalg.svd(A, full_matrices=False)
        s = s / (s * s + lam)
        return (VT.T * s) @ U.T
    except np.linalg.LinAlgError:
        pass
    try:
        return np.linalg.pinv(A, rcond=lam)
    except np.linalg.LinAlgError:
        pass
    try:
        X, *_ = np.linalg.lstsq(A, I_m, rcond=None)
        return X
    except np.linalg.LinAlgError:
        pass
    try:
        ATA = A.T @ A
        ridge = ATA + lam * np.eye(n, dtype=np.float64)
        return np.linalg.solve(ridge, A.T)
    except np.linalg.LinAlgError:
        return np.zeros((n, m), dtype=np.float64)

def _sieve_primes(n: int) -> List[int]:
    """
    Generate all prime numbers up to and including n using the Sieve of Eratosthenes.

    This is an efficient algorithm that marks non-primes by iteratively eliminating
    multiples of each found prime.

    Parameters
    ----------
    n : int
        Upper bound (inclusive) for prime number generation.

    Returns
    -------
    List[int]
        A list of all primes ≤ n, in increasing order.
    """
    sieve = np.ones(n+1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n**0.5) + 1):
        if sieve[p]:
            sieve[p*p : n+1 : p] = False
    return np.nonzero(sieve)[0].tolist()

@jit(nopython=True)
def mss_attack(report, moduli, k):
    """
    Recover a plausible original value from an obfuscated report using modular inversion.

    This simulates an attack where the adversary guesses the original input by
    enumerating all candidates that could result in the observed residue.

    Parameters
    ----------
    report : Tuple[int, int]
        A single obfuscated report (j, z), where:
        - j is the modulus index used,
        - z is the reported noisy modular value.

    Returns
    -------
    int
        An adversarial guess for the original value x ∈ [0, k).
    """
    j, Z = report
    m_j = moduli[j]

    x_candidates = List.empty_list(int64)
    for x in range(k):
        for z in Z:
            if x % m_j == z:
                x_candidates.append(x)
                break  # avoid duplicate matches if z repeats

    if len(x_candidates) == 0:
        return List.empty_list(int64)  

    return x_candidates

class ModularSubsetSelection(BaseProtocol):
    """
    Modular Subset Selection (MSS) Protocol for Local Differential Privacy.

    This protocol encodes categorical values into modular residues and applies
    Subset Selection (SS) independently per modulus, offering improved utility and
    flexibility via redundant, structured encodings. On the server side, estimates
    are obtained via generalized least squares with variance-optimal weights.

    Parameters
    ----------
    k : int
        Domain size of the data (number of categories).
    epsilon : float
        Local privacy budget.
    l : Optional[int], default=None
        Number of moduli to use. If None, selected analytically.
    l_max : int, default=50
        Maximum number of moduli for analytical search.
    lambda_reg : Optional[float], default=None
        Ridge regularization parameter used during estimation.
    seed : Optional[int], default=42
        Random seed for reproducibility.
    cache_moduli : bool, default=True
        Whether to cache and reuse moduli selections across runs.
    """

    def __init__(self, 
                 k: int, 
                 epsilon: float,
                 l: Optional[int] = None, 
                 l_max: int = 20,
                 lambda_reg: Optional[float] = None,
                 seed: Optional[int] = 42,
                 cache_moduli: bool = True) -> None:
        
        if not isinstance(k, int) or k < 2:
            raise ValueError("k must be an integer >= 2.")
        if epsilon <= 0:
            raise ValueError("epsilon must be greater than 0.")
        
        # Set initial parameters
        super().__init__(k, epsilon)        
        self.l_max = l_max
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.cache_moduli = cache_moduli
        self.lambda_reg = lambda_reg if lambda_reg is not None else 0 # 1.0 / epsilon**2
        
        # Setting moduli and weights
        self._Aw_dag = None        
        mods = (self._choose_moduli_analytical(l_max) if l is None 
                else self.find_valid_moduli(k, l, rng=self.rng, seed=self.seed, cache_moduli=self.cache_moduli))
        self.moduli = np.array(mods, dtype=int)
        self.l = len(self.moduli)

        # Parameters for Subset Selection
        omegas, p_list, q_list = [], [], []
        for m in self.moduli:
            ss = SubsetSelection(int(m), epsilon)
            omegas.append(ss.omega)
            p_list.append(ss.p)
            q_list.append(ss.q)
        self.omega = np.array(omegas)
        self._p_i = np.array(p_list)
        self._q_i = np.array(q_list)

        # Build the weighted design matrix A_w
        self._build_weighted_matrix()

    def obfuscate(self, data: Iterable[int]) -> List[Tuple[int, int]]:
        """
        Obfuscate a batch of input values using the MSS protocol.

        Each input value is mapped to a randomly chosen modular channel,
        where it is then obfuscated using the Subset Selection mechanism
        based on that modulus.

        Parameters
        ----------
        data : Iterable[int]
            Sequence of integer values in the domain {0, ..., k-1}.

        Returns
        -------
        List[Tuple[int, int]]
            A list of obfuscated reports in the form (j, z), where:
            - j is the selected modulus index
            - z is the obfuscated subset under that modulus
        """
        
        data = np.asarray(data, dtype=int)
        if np.any((data < 0) | (data >= self.k)):
            raise ValueError("One or more inputs are outside the domain.")
        
        j_arr = self.rng.integers(self.l, size=len(data))
        def obfuscate_single(x, j):
            return (
                j,
                ss_obfuscate(
                    x % self.moduli[j],
                    self.moduli[j],
                    self._p_i[j],
                    self.omega[j]
                )
            )
        results = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(obfuscate_single)(x, j) for x, j in zip(data, j_arr))
        
        return results

    def _debias_block(self, reports: Iterable[Tuple[int, int]]) -> np.ndarray:
        """
        Compute the unbiased sufficient statistics vector from obfuscated reports.

        Each report (j, z) is counted within its modulus group. For each modulus,
        an empirical histogram is computed and de-biased using the known parameters
        of the corresponding Subset Selection channel.

        The final output is the concatenated and weighted vector of all unbiased
        block-level statistics, scaled by the same `w_sqrt` used in the GLS system.

        Parameters
        ----------
        reports : Iterable[Tuple[int, int]]
            Obfuscated user reports, where j is the modulus index, and z is the reported value.

        Returns
        -------
        np.ndarray
            A 1D vector of length sum(m_i) containing the de-biased, weighted sufficient statistics.
        """

        counts = [np.zeros(m, float) for m in self.moduli]
        n_j    = np.zeros(self.l, int)
        for j, z in reports:
            counts[j][z] += 1
            n_j[j]       += 1

        s_blocks = []
        for j in range(self.l):
            p, q   = self._p_i[j], self._q_i[j]
            pq     = p - q
            hist   = counts[j] / n_j[j]          # empirical probs
            s_hat  = (hist - q) / pq             # unbiased Bernoulli estimator
            s_blocks.append(s_hat)

        # multiply by exact same w_sqrt you used in A_w
        s_vec = np.concatenate(s_blocks) * self._w_sqrt
        return s_vec

    # all three methods below are alternatives for estimating the raw frequency vector

    def estimate_raw(self, reports: Iterable[Tuple[int, int]]) -> np.ndarray:
        """
        Estimate the raw (unnormalized) frequency vector from reports.

        Solves a regularized least-squares problem using LSMR:
            min_x ||A_w x - s||^2 + λ ||x||^2

        where A_w is the weighted design matrix and s is the de-biased statistics vector.

        Parameters
        ----------
        reports : Iterable[Tuple[int, int]]
            List of obfuscated (j, z) reports from users.

        Returns
        -------
        np.ndarray
            The estimated raw frequency vector (not necessarily normalized or non-negative).
        """

        s_vec= self._debias_block(reports)

        freq_estimates = lsmr(self._Aw, s_vec, damp=np.sqrt(self.lambda_reg))[0]
        return freq_estimates

    # def estimate_raw2(self, reports: Iterable[Tuple[int, int]]) -> np.ndarray:
    #     s_vec = self._debias_block(reports)

    #     # Ensure cached pseudo-inverse exists
    #     if self._Aw_dag is None or self._Aw_dag.shape[1] != self._Aw.shape[0]:
    #         self._Aw_dag = _safe_sparse_pinv(self._Aw, lam=self.lambda_reg)

    #     # Estimate directly using pseudo-inverse
    #     freq_estimates = self._Aw_dag @ s_vec
    #     return freq_estimates

    # def estimate_raw3(self, reports: Iterable[Tuple[int, int]]) -> np.ndarray:
    #     s_vec = self._debias_block(reports)

    #     A = self._Aw.toarray() if sp.issparse(self._Aw) else self._Aw
    #     ATA = A.T @ A
    #     ridge = ATA + self.lambda_reg * np.eye(self.k)
    #     freq_estimates = np.linalg.solve(ridge, A.T @ s_vec)

    #     return freq_estimates
    
    def estimate(self, reports: Iterable[Tuple[int, int]]) -> np.ndarray:
        
        """
        Estimate the original distribution from obfuscated reports.

        This method applies de-biasing and solves a regularized least-squares
        problem to estimate the frequency vector. The result is re-normalized
        to ensure it sums to 1 (or projected to the probability simplex),
        to ensure a valid distribution.

        Parameters
        ----------
        reports : Iterable[Tuple[int, int]]
            Obfuscated reports (j, z) from clients.

        Returns
        -------
        np.ndarray
            Normalized frequency vector summing to 1.
        """

        x_hat = self.estimate_raw(reports)
        x_hat = np.maximum(x_hat, 0.0)  # ensure non-negativity
        x_hat = x_hat / x_hat.sum()  # normalize to sum to 1

        return x_hat
        # Alternative: 
        # return self._project_to_simplex(freq_estimates)
    
    @staticmethod
    def _project_to_simplex(v: np.ndarray) -> np.ndarray:
        """
        Return the Euclidean projection of v onto the probability simplex

            Δ_{k-1} = { x ∈ ℝ^k : x ≥ 0,  ∑_i x_i = 1 }.

        Uses the O(k log k) algorithm by Michelot (1986) / Duchi et al. (2008).
        """
        if v.ndim != 1:
            raise ValueError("input must be a 1-D vector")
        k = v.size
        # Sort v in descending order
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - 1.0
        rho  = np.nonzero(u - cssv / (np.arange(k) + 1) > 0)[0][-1]
        theta = cssv[rho] / (rho + 1)
        w = v - theta
        w[w < 0.0] = 0.0
        return w
    
    def _approx_mse(self, n: int = 1, probes: int = 32) -> float:
        """
        Monte-Carlo estimate of the analytical MSE that avoids materialising
        the dense pseudo-inverse.  Accuracy improves ~1/√probes.

        Parameters
        ----------
        n       : sample size (affects Σ scaling)
        probes  : number of Hutchinson probe vectors (≥16 recommended)

        Returns
        -------
        float
        """

        # --- row-wise std-devs of Σ 
        inv_nj  = self.l / n
        row_std = []
        for j, m in enumerate(self.moduli):
            p, q   = self._p_i[j], self._q_i[j]
            pq     = p - q
            pi     = q + pq / m
            var    = pi * (1.0 - pi) * inv_nj / pq**2
            row_std.extend([np.sqrt(var)] * m)
        row_std = np.asarray(row_std, dtype=np.float64)

        λ_sqrt  = np.sqrt(self.lambda_reg)
        acc     = 0.0

        for _ in range(probes):
            z  = self.rng.standard_normal(self._Aw.shape[0])
            v  = self._w_sqrt * (row_std * z)              # Σ^{½} D_w z
            x  = lsmr(self._Aw, v, damp=λ_sqrt,
                      atol=1e-6, btol=1e-6, conlim=1e8)[0]  # (AwᵀAw+λI)⁻¹Awᵀ v
            acc += np.dot(x, x)

        return (acc / probes) / self.k

    def get_mse(self, n: int = 1, *,
                approx_threshold: int = 0, # fallback for fast approximation
                probes: int = 32) -> float:
        """
        Analytical MSE.  Uses the Hutch++ approximation.

        Parameters
        ----------
        n               : sample size
        approx_threshold: switch to fast path once k > threshold
        probes          : number of Hutchinson probe vectors

        Returns
        -------
        float
        """
        # large domain → Hutch++ fast path 
        if self.k > approx_threshold:
            return self._approx_mse(n=n, probes=probes)

        # original exact path 
        if self._Aw.shape[0] != len(self._w_sqrt):
            raise ValueError("Mismatch between A and weights")
        if self._Aw_dag is None or self._Aw_dag.shape[1] != self._Aw.shape[0]:
            self._Aw_dag = _safe_sparse_pinv(self._Aw, lam=self.lambda_reg)

        inv_nj = self.l / n
        blocks = []
        for j, m in enumerate(self.moduli):
            p, q  = self._p_i[j], self._q_i[j]
            pq    = p - q
            pi    = q + pq / m
            cov_y = (np.eye(m)*(pi*(1-pi)) - np.full((m, m), pi**2)) * inv_nj
            blocks.append(cov_y / pq**2)
        Sigma = scipy.linalg.block_diag(*blocks)
        G     = self._Aw_dag @ np.diag(self._w_sqrt)
        return float(np.trace(G @ Sigma @ G.T) / self.k)
    
    def _build_weighted_matrix(self) -> None:
        """
        Construct the design matrix A_w with variance-optimal weights.

        Each modulus j defines a Subset Selection channel over a residue system.
        This function builds a vertical stack of sparse matrices A_j corresponding
        to each modulus, where each row is scaled by:
            w_j = pq / sqrt(π (1 - π))
        with pq = p - q and π the probability of a 1-entry in the Bernoulli channel.

        The resulting matrix A_w has:
            - shape (sum(m_j), k)
            - row i scaled by 1 / σ_i, the inverse standard deviation

        This ensures isotropic noise in the least-squares decoding, so that
        Generalized Least Squares reduces to Ordinary Least Squares.

        Sets
        ----
        self._Aw : scipy.sparse.csr_matrix
            The stacked and weighted design matrix.
        self._w_sqrt : np.ndarray
            The vector of weights applied to each row (used for both A_w and debiasing).
        """

        k       = self.k
        x_vals  = np.arange(k, dtype=np.int32)
        rows    = []
        w_sqrt  = []

        for j, m in enumerate(self.moduli):
            p, q    = self._p_i[j], self._q_i[j]
            pq      = p - q
            pi      = q + pq / m                 # prob. that a row-entry is 1
            var     = pi * (1.0 - pi)            # Bernoulli variance
            sqrt_w  = pq / np.sqrt(var)        # == 1 / σ_i

            residue = x_vals % m                 # vectorised modulo
            data    = np.full(k, sqrt_w, dtype=np.float64)
            A_j     = csr_matrix((data, (residue, x_vals)), shape=(m, k))

            rows.append(A_j)
            w_sqrt.extend([sqrt_w] * m)

        self._Aw      = vstack(rows).tocsr()
        self._w_sqrt  = np.asarray(w_sqrt, dtype=np.float64)
        self._Aw_dag  = None          # invalidate cached pseudo-inverse


    def _lower_bound_ell(self, 
                        k: int,
                        eps: float,
                        kappa_max: float = 1e1,
                        beta: float = 2e1,
                        ) -> int:
        """
        Return the theoretical minimum block count ell_theory
        such that ell >= g(ell) holds.

        Parameters
        ----------
        k : int
            Domain size.
        eps : float
            Privacy budget ε.
        kappa_max : float, optional
            Target condition number κ_max.
        beta : float, optional
            User-selected search width b.

        Returns
        -------
        int
            The smallest admissible ell_theory.
        """
        alpha = (beta + np.exp(eps)) / ((1 / beta) + np.exp(eps))

        for ell in range(2, self.l_max + 1):
            t_star = np.ceil(np.log(k) / np.log(k / (beta * ell)))
            g_ell  = (1 + kappa_max / alpha) / (kappa_max / alpha - 1) * t_star
            if ell >= g_ell:
                return int(ell)

        raise ValueError("l_max too small – enlarge search range.")


    def _choose_moduli_analytical(self, l_max: int,
                                    trials: int = 1_000,
                                    kappa_max: float = 1e1,
                                    beta: float = 2e1,
                                    ) -> List[int]:
        """
        Select an optimal set of moduli analytically by minimizing the estimated MSE.

        Iterates over candidate block counts ℓ ∈ [2, l_max], trying random modulus tuples,
        and selects the one yielding the lowest analytical MSE estimate.

        Parameters
        ----------
        l_max : int
            Maximum number of modulus blocks to consider.
        trials : int, default=1_000
            Number of random modulus sets to try for each block count.
        kappa_max : float, default=10
            Target maximum condition number for the system matrix.
        beta : float, default=20
            Controls the width of the modulus selection band.

        Returns
        -------
        List[int]
            Optimal list of prime moduli minimizing the expected MSE.

        Raises
        ------
        ValueError
            If no valid moduli are found within the search space.
        """
        
        if self.cache_moduli:
            mod_cache = load_mod_cache()
            key = get_mod_cache_key(self.k, self.epsilon, self.l_max, seed=self.seed, prefix="opt")
            if key in mod_cache:
                return mod_cache[key]
                    
        # ell_theory = self._lower_bound_ell(self.k, self.epsilon, kappa_max, beta)
        # if ell_theory is None:
        ell_theory = 2  # fallback to minimum block count via Remark 2

        best_mse, best_sel = np.inf, None
        for ell in range(ell_theory, l_max+1):
            try:
                mods = self.find_valid_moduli(k=self.k, l=ell, trials=trials, kappa_max=kappa_max, beta=beta, rng=self.rng, cache_moduli=self.cache_moduli)
            except ValueError:
                continue
            self.moduli = np.array(mods, int)
            self.l = ell
            p_list, q_list = [], []
            for m in self.moduli:
                ss = SubsetSelection(int(m), self.epsilon)
                p_list.append(ss.p)
                q_list.append(ss.q)
            self._p_i = np.array(p_list)
            self._q_i = np.array(q_list)
            self._build_weighted_matrix()
            mse = self.get_mse(approx_threshold=0)
            if mse < best_mse:
                best_mse, best_sel = mse, mods
        if best_sel is None:
            raise ValueError("no moduli found within l_max")

        # Save best moduli under the 'opt:' cache key
        if self.cache_moduli:
            mod_cache[key] = best_sel
            save_mod_cache(mod_cache)
        return best_sel
    
    @classmethod
    def find_valid_moduli(
        cls,
        k: int,
        l: int,
        *,
        trials: int = 1_000,
        kappa_max: float = 1e1,
        beta: float = 2e1,
        rng: Optional[np.random.Generator] = None,
        seed: int | None = None,
        cache_moduli: bool = True
    ) -> List[int]:
        """
        Heuristically find a valid set of moduli for CRT encoding.

        Searches for l primes within a controlled band such that:
        - Their product is ≥ k (CRT covers the domain),
        - Their total number of degrees of freedom (∑(m_i - 1)) ≥ k (full-rank),
        - The condition number of the resulting Aw matrix is ≤ kappa_max.

        Parameters
        ----------
        k : int
            Domain size.
        l : int
            Number of moduli to select.
        trials : int, default=1_000
            Number of random samples tried.
        kappa_max : float, default=10
            Maximum acceptable condition number.
        beta : float, default=20
            Bandwidth parameter for modulus sampling.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        cache_moduli : bool, default=True
            Whether to reuse moduli from cache.

        Returns
        -------
        List[int]
            A valid set of moduli.

        Raises
        ------
        ValueError
            If no valid tuple is found within `trials`.
        """
        
        if rng is None:
            rng = np.random.default_rng(seed)

        if cache_moduli:
            mod_cache = load_mod_cache()
            key = get_mod_cache_key(k, l_max=l, seed=seed, prefix="raw")  # or f"{k}:{l}"

            if key in mod_cache:
                return list(map(int, mod_cache[key]))

        # helpers
        def _is_prime(n: int) -> bool:
            if n < 2:
                return False
            if n in (2, 3):
                return True
            if n % 2 == 0:
                return False
            lim = int(n ** 0.5) + 1
            for d in range(3, lim, 2):
                if n % d == 0:
                    return False
            return True
        
        def _next_prime(n: int) -> int:
            if n <= 2:
                return 2
            p = n + (n & 1 == 0)
            while True:
                if all(p % d for d in range(3, int(p**0.5) + 1, 2)):
                    return p
                p += 2

        def _cond(mods: List[int]) -> float:
            """
            Return an estimate of cond(Aw).  
            """

            dummy = cls.__new__(cls)
            dummy.k      = k
            dummy.l      = len(mods)
            dummy.moduli = np.array(mods, int)
            dummy.lambda_reg = 1e-6
            dummy.eps_vec    = np.ones(dummy.l)
            dummy._p_i = np.full(dummy.l, 1 / (1 + mods[0]))
            dummy._q_i = np.full(dummy.l, 1 / mods[0])
            dummy._build_weighted_matrix()
            s = svds(dummy._Aw, k=2, tol=1e-3,
                     return_singular_vectors=False)
            return s.max() / s.min()

        # sampling band
        target = max(3, k // l)               # centre ≈ k/ℓ
        L     = max(3, int(target / beta))
        H     = max(L + 2, int(target * beta))
        H     = min(H, int(0.95 * k))       
        
        primes = [p for p in _sieve_primes(H + 1000) if L <= p <= H]

        if len(primes) < l:
            raise ValueError("band too narrow – widen beta.")

        best, best_cond = None, np.inf

        for trial in range(trials):
            mods = rng.choice(primes, l, replace=False).tolist()
            mods.sort()

            # ensure CRT coverage and full rank
            while np.prod(mods) < k or sum(m - 1 for m in mods) < k:
                j = rng.integers(l)
                mods[j] = _next_prime(mods[j] + 1)
            
            # Check condition number
            c = _cond(mods)
            if c < best_cond:
                best, best_cond = mods.copy(), c
                if best_cond <= kappa_max:# and trial > 1:
                    break  # good enough – stop early
                
        # final check
        if best is None or best_cond > kappa_max:
            for offset in range(20):
                base = int(np.ceil(k ** (1.0 / l))) + offset
                mods = []
                p = base
                while len(mods) < l:
                    if _is_prime(p):
                        mods.append(p)
                    p += 1

                idx = 0
                while np.prod(mods) < k or sum(m - 1 for m in mods) < k:
                    mods[idx] = _next_prime(mods[idx] + 1)
                    idx = (idx + 1) % l

                # Reject duplicate moduli 
                if len(set(mods)) < l:
                    continue

                cond = _cond(mods)
                if cond <= kappa_max:
                    best = mods
                    best_cond = cond
                    break
        if best is None or best_cond > kappa_max:
            raise ValueError("no valid moduli found within trials.")
        
        if cache_moduli:
            mod_cache[key] = list(map(int, best))
            save_mod_cache(mod_cache)
        
        return best

    def attack(self, report: Tuple[int, int]) -> int:
        """
        Recover a plausible original value from an obfuscated report using modular inversion.

        This simulates an attack where the adversary guesses the original input by
        enumerating all candidates that could result in the observed residue.

        Parameters
        ----------
        report : Tuple[int, int]
            A single obfuscated report (j, z), where:
            - j is the modulus index used,
            - z is the reported noisy modular value.

        Returns
        -------
        int
            An adversarial guess for the original value x ∈ [0, k).
        """
        candidates = mss_attack(report, self.moduli, self.k)
        return self.rng.choice(candidates)

    def get_asr(self) -> float:
        """
        Compute the theoretical Attacker Success Rate (ASR) under a uniform prior.

        The ASR represents the expected probability that a uniformly random input
        can be correctly guessed by an attacker using modular inversion and
        knowledge of the protocol parameters.

        Returns
        -------
        float
            The average attacker success rate over all inputs.
        """

        total = 0.0
        for j in range(self.l):
            m_j = self.moduli[j]
            omega_j = self.omega[j]
            num = m_j * np.exp(self.epsilon)
            denom = omega_j*np.exp(self.epsilon) + m_j - omega_j
            total += num / denom
        return total / (self.l * self.k)

    @property
    def design_matrix(self):
        return self._Aw.copy()

    @property
    def weights(self):
        return self._w_sqrt.copy()
