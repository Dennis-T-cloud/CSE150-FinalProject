# hmm_utils.py
import numpy as np
# -------------------------------------------------------------------
# Core HMM pieces (Dennis update:)
# These functions are basically the same style as the HMM mini-project.
# plz free feel modify the main logic and just tweak shapes / names if needed.
# -------------------------------------------------------------------

# normalize in log-space (log-sum-exp) to keep numbers sane
def logsumexp(vec):
    m = np.max(vec)
    return m + np.log(np.sum(np.exp(vec - m)))

def initialize_hmm_params(K, observations, seed=None):
    """
    Simple helper to initialize HMM parameters.

    - K: number of hidden states
    - observations: 1D numpy array of y_t (e.g., log-returns)
    - seed: random seed for reproducibility

    Returns a dict with:
        pi    : (K,) initial state distribution
        A     : (K, K) transition matrix
        means : (K,) Gaussian means
        vars  : (K,) Gaussian variances (diagonal covariance)
    """
    rng = np.random.default_rng(seed)

    # init pi as nearly uniform
    pi = np.ones(K) / K

    # init A with random rows, normalized
    A = rng.random((K, K))
    A = A / A.sum(axis=1, keepdims=True)

    # init Gaussian params based on data
    data_mean = np.mean(observations)
    data_std = np.std(observations) + 1e-6

    means = data_mean + rng.normal(scale=data_std, size=K)
    vars_ = np.full(K, data_std**2)

    params = {
        "pi": pi,
        "A": A,
        "means": means,
        "vars": vars_
    }
    return params


def gaussian_logpdf(x, mean, var):
    """
    Log density of 1D Gaussian N(mean, var) for value x.

    This is tiny and simple, but we keep it here to avoid repeating code.
    """
    # avoid numerical issues
    var = max(var, 1e-12)
    return -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var)


def forward_pass(observations, params):
    """
    Forward algorithm (alpha recursion) in log-space.

    This is basically the same as the HMM mini-project forward function.
    You can paste your existing implementation here if you prefer.

    Returns:
        log_alpha: (T, K) log alpha_t(k)
        log_likelihood: scalar log P(y_1:T)
    """
    # TODO: replace with your mini-project implementation if you like.
    pi = params["pi"]
    A = params["A"]
    means = params["means"]
    vars_ = params["vars"]

    T = len(observations)
    K = len(pi)

    log_alpha = np.zeros((T, K))
    # log emission at t for each state
    log_emiss = np.zeros(K)

    # init step
    for k in range(K):
        log_emiss[k] = gaussian_logpdf(observations[0], means[k], vars_[k])
        log_alpha[0, k] = np.log(pi[k] + 1e-32) + log_emiss[k]

    

    # recursion
    for t in range(1, T):
        for k in range(K):
            for_prev = log_alpha[t - 1] + np.log(A[:, k] + 1e-32)
            log_trans = logsumexp(for_prev)
            log_emiss[k] = gaussian_logpdf(observations[t], means[k], vars_[k])
            log_alpha[t, k] = log_trans + log_emiss[k]

    log_likelihood = logsumexp(log_alpha[-1])
    return log_alpha, log_likelihood


def backward_pass(observations, params):
    """
    Backward algorithm (beta recursion) in log-space.

    This is also very similar to the HMM mini-project version.

    Returns:
        log_beta: (T, K) log beta_t(k)
    """
    pi = params["pi"]
    A = params["A"]
    means = params["means"]
    vars_ = params["vars"]

    T = len(observations)
    K = len(pi)

    log_beta = np.zeros((T, K))

    # init: log_beta_T(k) = 0 for all k
    log_beta[T - 1, :] = 0.0

    # recursion backwards
    for t in range(T - 2, -1, -1):
        for i in range(K):
            terms = np.zeros(K)
            for j in range(K):
                log_emiss = gaussian_logpdf(observations[t + 1], means[j], vars_[j])
                terms[j] = np.log(A[i, j] + 1e-32) + log_emiss + log_beta[t + 1, j]
            log_beta[t, i] = logsumexp(terms)

    return log_beta


def compute_posteriors(log_alpha, log_beta):
    """
    Compute gamma_t(k) = P(S_t = k | y_1:T) from log_alpha and log_beta.

    This is standard mini-project style too.
    """
    T, K = log_alpha.shape
    log_gamma = log_alpha + log_beta

    # normalize each time slice using log-sum-exp
    for t in range(T):
        log_norm = logsumexp(log_gamma[t])
        log_gamma[t] = log_gamma[t] - log_norm
    gamma = np.exp(log_gamma)
    return gamma


def baum_welch_train(observations, K, max_iters=100, tol=1e-6, n_restarts=3, seed=None):
    """
    Run EM / Baum-Welch to estimate HMM parameters.

    This is your main "learning" function, similar to the mini-project.
    Feel free to copy in your cleanest version and adapt the emission update.

    Returns:
        best_params: dict with pi, A, means, vars
        best_loglik_trace: list of log-likelihood values per iteration
    """
    rng = np.random.default_rng(seed)

    best_params = None
    best_trace = None
    best_final_ll = -np.inf

    for restart in range(n_restarts):
        # new seed for each restart
        restart_seed = rng.integers(0, 10_000)
        params = initialize_hmm_params(K, observations, seed=restart_seed)

        loglik_trace = []
        prev_ll = None

        for it in range(max_iters):
            # E-step: forward-backward
            log_alpha, loglik = forward_pass(observations, params)
            log_beta = backward_pass(observations, params)
            gamma = compute_posteriors(log_alpha, log_beta)

            T = len(observations)
            K = len(params["pi"])

            # xi_t(i,j): pairwise posteriors
            xi = np.zeros((T - 1, K, K))

            for t in range(T - 1):
                for i in range(K):
                    for j in range(K):
                        log_emiss = gaussian_logpdf(observations[t + 1],
                                                    params["means"][j],
                                                    params["vars"][j])
                        xi[t, i, j] = (
                            log_alpha[t, i]
                            + np.log(params["A"][i, j] + 1e-32)
                            + log_emiss
                            + log_beta[t + 1, j]
                        )
                # normalize slice in probability space
                # (we do it in log-space then exponentiate)
                log_norm = logsumexp(xi[t].reshape(-1))
                xi[t] = np.exp(xi[t] - log_norm)

            # M-step: update pi, A, Gaussian params
            gamma_sum = gamma.sum(axis=0)  # shape (K,)

            # update pi
            # add small constant to avoid div by zero
            params["pi"] = gamma[0] / (gamma[0].sum() + 1e-32)

            # update A
            # add small constant to avoid div by zero
            xi_sum = xi.sum(axis=0)  # shape (K, K)
            params["A"] = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + 1e-32)

            # update means and vars
            new_means = np.zeros(K)
            new_vars = np.zeros(K)
            for k in range(K):
                weights = gamma[:, k]
                weighted_sum = np.sum(weights * observations)
                new_means[k] = weighted_sum / (gamma_sum[k] + 1e-32)

                diff = observations - new_means[k]
                weighted_var = np.sum(weights * diff**2) / (gamma_sum[k] + 1e-32)
                new_vars[k] = max(weighted_var, 1e-12)

            params["means"] = new_means
            params["vars"] = new_vars

            loglik_trace.append(loglik)

            # simple convergence check
            if prev_ll is not None and abs(loglik - prev_ll) < tol:
                break
            prev_ll = loglik

        # keep the best restart
        final_ll = loglik_trace[-1]
        if final_ll > best_final_ll:
            best_final_ll = final_ll
            best_params = params
            best_trace = loglik_trace

    return best_params, best_trace


def viterbi_decode(observations, params):
    """
    Viterbi algorithm to get the most likely hidden state sequence.

    Again, same spirit as the mini-project Viterbi code.
    Returns:
        path: (T,) int array of state indices (0..K-1)
    """
    pi = params["pi"]
    A = params["A"]
    means = params["means"]
    vars_ = params["vars"]

    T = len(observations)
    K = len(pi)

    log_delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)

    # init
    for k in range(K):
        log_delta[0, k] = np.log(pi[k] + 1e-32) + gaussian_logpdf(observations[0], means[k], vars_[k])
        psi[0, k] = 0

    # recursion
    for t in range(1, T):
        for j in range(K):
            # previous state i that maximizes
            scores = log_delta[t - 1] + np.log(A[:, j] + 1e-32)
            best_prev = np.argmax(scores)
            log_delta[t, j] = scores[best_prev] + gaussian_logpdf(observations[t], means[j], vars_[j])
            psi[t, j] = best_prev

    # backtrack
    path = np.zeros(T, dtype=int)
    path[T - 1] = np.argmax(log_delta[T - 1])
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path


# -------------------------------------------------------------------
# Model selection helpers
# -------------------------------------------------------------------

def count_num_params(K):
    """
    Rough count of free parameters in this HMM:
    - pi: K-1 free params (since they sum to 1)
    - A: K*(K-1) free params
    - means: K
    - vars: K

    You can adjust if you change the model.
    """
    pi_params = K - 1
    A_params = K * (K - 1)
    mean_params = K
    var_params = K
    return pi_params + A_params + mean_params + var_params


def compute_aic_bic(loglik, num_params, T):
    """
    Compute AIC and BIC given log-likelihood, number of parameters, and T.

    AIC = -2 * log L + 2p
    BIC = -2 * log L + p * log(T)
    """
    aic = -2 * loglik + 2 * num_params
    bic = -2 * loglik + num_params * np.log(T)
    return aic, bic
