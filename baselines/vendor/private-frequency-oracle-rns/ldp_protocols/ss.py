import numpy as np
from numba import jit
from ldp_protocols.base import BaseProtocol
from typing import Iterable, List

@jit(nopython=True)
def ss_obfuscate(input_data: int, k: int, p: float, omega: int) -> np.ndarray:
    """
    Obfuscate the input data using the Subset Selection (SS) protocol.

    Parameters
    ----------
    input_data : int
        The user's true value to be obfuscated. Must be in the range [0, k-1].
    k : int
        The size of the attribute's domain. Must be an integer >= 2.
    p : float
        The probability of including the true input value in the sanitized subset. 
        Derived from the privacy budget epsilon.
    omega : int
        The size of the output subset (number of values in the obfuscated result).

    Returns
    -------
    numpy.ndarray
        A sanitized subset (array) of size `omega` containing obfuscated values.

    Raises
    ------
    ValueError
        If `input_data` is not in the range [0, k-1].
    """
    if input_data < 0 or input_data >= k:
        raise ValueError("input_data must be in the range [0, k-1].")

    # Mapping domain size k to the range [0, ..., k-1]
    domain = np.arange(k)
    omega= int(omega)
    # SS perturbation function
    sub_set = np.zeros(omega, dtype='int64')
    if np.random.random() <= p:
        sub_set[0] = int(input_data)
        sub_set[1:] = np.random.choice(domain[domain != input_data], size=omega - 1, replace=False)
        np.random.shuffle(sub_set)
        return sub_set.astype(np.int64)
    else:
        return np.random.choice(domain[domain != input_data], size=omega, replace=False).astype(np.int64)

@jit(nopython=True)
def attack_ss(obfuscated_vec: np.ndarray) -> int:
    """
    Perform a privacy attack on an obfuscated subset generated using the Subset Selection (SS) protocol.

    This method attempts to infer the true value by randomly selecting a value from the obfuscated subset.
    Since the true value is included with higher probability in the subset, an adversary can exploit this 
    to make an educated guess.

    Parameters
    ----------
    obfuscated_vec : np.ndarray
        An obfuscated subset of values generated using the SS protocol. 
        The subset contains a fixed number of values selected from the domain.

    Returns
    -------
    int
        The inferred true value of the input. This is selected randomly from the values present 
        in the obfuscated subset.
    """
                
    return np.random.choice(obfuscated_vec)

@jit(nopython=True)
def ss_asr(epsilon: float, k: int, omega: int) -> float:
    """
    Calculate the Adversarial Success Rate (ASR) of the Subset Selection (SS) protocol.

    Parameters
    ----------
    epsilon : float
        The privacy budget for the SS mechanism. Must be a positive value.
    k : int
        The size of the attribute's domain. Must be an integer >= 2.
    omega : int
        The size of the output subset.

    Returns
    -------
    float
        The Adversarial Success Rate (ASR), which quantifies the adversary's ability 
        to correctly guess the true value from the sanitized subset.
    """
    # Calculate parameters
    return np.exp(epsilon) / (omega * np.exp(epsilon) + k - omega)

class SubsetSelection(BaseProtocol):
    def __init__(self, k: int, epsilon: float):
        """
        Initialize the Subset Selection (SS) protocol.

        Parameters
        ----------
        k : int
            The size of the attribute's domain. Must be an integer >= 2.
        epsilon : float
            The privacy budget for the SS mechanism. Must be a positive value.

        Raises
        ------
        ValueError
            If `k` is not an integer >= 2 or if `epsilon` is not a positive value.
        """
        if not isinstance(k, int) or k < 2:
            raise ValueError("k must be an integer >= 2.")
        if epsilon <= 0:
            raise ValueError("epsilon must be a numerical value greater than 0.")
        
        super().__init__(k, epsilon)
        self.omega = int(max(1, np.rint(self.k / (np.exp(self.epsilon) + 1))))
        self.p = (self.omega * np.exp(self.epsilon)) / (self.omega * np.exp(self.epsilon) + self.k - self.omega)
        self.q = (self.omega * np.exp(self.epsilon) * (self.omega - 1) + (self.k - self.omega) * self.omega) / ((self.k - 1) * (self.omega * np.exp(self.epsilon) + self.k - self.omega))

    def obfuscate(self, input_data: Iterable[int]) -> List[np.ndarray]:
        """
        Obfuscate multiple input data points using the Subset Selection (SS) mechanism.

        Parameters
        ----------
        input_data : Iterable[int]
            Iterable of user true input values to be sanitized.
            Each must be in the range [0, k-1].

        Returns
        -------
        List[np.ndarray]
            A list of sanitized subsets (arrays), each of size `omega`.
        """
        data = np.asarray(input_data, dtype=int)
        if np.any((data < 0) | (data >= self.k)):
            raise ValueError("One or more inputs are outside the domain.")

        # Vectorized version: uses list comprehension but could be Parallel too
        return [
            ss_obfuscate(x, self.k, self.p, self.omega)
            for x in data
        ]
    
    def estimate_raw(self, noisy_reports: list) -> np.ndarray:
        """
        Estimate frequencies from noisy reports collected using the Subset Selection (SS) mechanism.

        This method applies unbiased estimation to the collected noisy reports to approximate 
        the true frequencies of values in the domain. It uses SS-specific parameters `p` (true value probability)
        and `q` (false value probability) to correct for the randomized responses.

        Parameters
        ----------
        noisy_reports : list of int
            A list of noisy reports collected from users. Each report corresponds to a single obfuscated value
            within the domain `[0, k-1]`, chosen as part of a subset generated by the SS mechanism.

        Returns
        -------
        np.ndarray
            An array of estimated frequencies for each value in the domain.
            The output array has size `k` and sums to 1.

        Raises
        ------
        ValueError
            If `noisy_reports` is empty.
        """
        n = len(noisy_reports)  # Number of reports
        if n == 0:
            raise ValueError("Noisy reports cannot be empty.")
        
        # Count the occurrences of each value in the noisy reports
        support_counts = np.zeros(self.k)
        for report in noisy_reports:
            support_counts[report] += 1

        # Unbiased frequency estimation
        freq_estimates = (support_counts - n * self.q) / (n * (self.p - self.q))
        
        return freq_estimates
    
    def estimate(self, noisy_reports: list) -> np.ndarray:
        """
        Return a normalized estimate of frequencies for each domain value,
        ensuring a valid probability vector that sums to 1.

        This method first computes the unbiased raw estimates (via `estimate_raw`),
        then clips negative values to zero and renormalizes the vector.

        Parameters
        ----------
        noisy_reports : list of int
            List of obfuscated reports from users.

        Returns
        -------
        np.ndarray
            Estimated frequency distribution over the domain.
            The output sums to 1 and all entries are non-negative.
        """
        freq_estimates = np.maximum(self.estimate_raw(noisy_reports), 0.0)
        return freq_estimates / np.sum(freq_estimates)
        # Alternative: self._project_to_simplex(freq_estimates)
    
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

    def attack(self, obfuscated_vec: np.ndarray) -> int:
        """
        Perform a privacy attack on an obfuscated subset generated using the Subset Selection (SS) protocol.

        Parameters
        ----------
        obfuscated_vec : np.ndarray
            An obfuscated subset of values generated using the SS protocol. 
            The subset contains a fixed number of values selected from the domain.

        Returns
        -------
        int
            The inferred true value of the input. This is selected randomly from the values present 
        in the obfuscated subset.

        Raises
        ------
        ValueError
            If `obfuscated_vec` is empty.
        """
        if len(obfuscated_vec) == 0:
            raise ValueError("Obfuscated subset cannot be empty")

        return attack_ss(obfuscated_vec)

    def get_mse(self, n: int = 1) -> float:
        """
        Compute the MSE of the Subset Selection (SS) protocol.

        Returns
        -------
        float
            The MSE of the SS mechanism, quantifying the expected estimation error.
        """

        return self.q * (1 - self.q) / (n * (self.p - self.q)**2)
    
    def get_asr(self) -> float:
        """
        Compute the Adversarial Success Rate (ASR) of the Subset Selection (SS) protocol.

        Returns
        -------
        float
            The Adversarial Success Rate (ASR), representing the adversary's ability to 
            correctly guess the true value.
        """
        return ss_asr(self.epsilon, self.k, self.omega)
