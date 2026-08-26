"""Phase 7: on-demand classical ML tool. Four parameterized scikit-learn
functions — the model (LLM or a direct API caller) supplies data and
hyperparameters, scikit-learn always does the actual fitting/scoring. This
is the same "orchestrator picks tool + args, model interprets the result,
doesn't compute it" contract already used throughout app/agent.py, applied
here to statistics instead of arithmetic: never let an LLM eyeball a scatter
plot and claim an R², always fit a real model and report its real score.

Guardrails (per the explicit "random queries won't execute" requirement):
every function validates its inputs before touching scikit-learn — shape
mismatches, too few samples for a meaningful train/test split, non-finite
values, and out-of-range hyperparameters (n_clusters/n_components larger
than the data supports) all raise MLToolError instead of running, so a
malformed or nonsensical call fails clearly rather than producing a
meaningless or crashing fit.
"""

import numpy as np
import scipy.stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.model_selection import train_test_split

MIN_SAMPLES = 10  # below this, a train/test split isn't meaningful


class MLToolError(Exception):
    """Raised when input fails validation — the tool refuses to run rather
    than fitting on garbage data."""


def _to_array(name: str, values, min_dims: int = 1, max_dims: int = 2) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MLToolError(f"{name} must be numeric: {exc}") from exc
    if arr.ndim < min_dims or arr.ndim > max_dims:
        raise MLToolError(f"{name} must have {min_dims}-{max_dims} dimensions, got {arr.ndim}")
    if arr.size == 0:
        raise MLToolError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise MLToolError(f"{name} contains NaN/Inf values")
    return arr


def _check_min_samples(n: int) -> None:
    if n < MIN_SAMPLES:
        raise MLToolError(f"Need at least {MIN_SAMPLES} samples for a meaningful train/test split, got {n}")


def _to_feature_matrix(name: str, values) -> np.ndarray:
    """X validated as a samples x features matrix. A flat list of scalars —
    the natural shape for "fit y against this one x variable", and what a
    caller (LLM or otherwise) reaches for by default — is reshaped to an
    (n, 1) column vector instead of rejected; sklearn only accepts 2D, but
    there's no reason the caller needs to know that convention just to fit a
    single-variable regression. Caught live: the agent's own fit_linear_regression
    call failed outright on a flat X (Pydantic rejected it before this
    function's body even ran, so the MLToolError path never got a chance to
    give a useful message) — fixed at both layers (see app/agent.py's tool
    signatures)."""
    arr = _to_array(name, values, min_dims=1, max_dims=2)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def fit_linear_regression(
    X: list[list[float]],
    y: list[float],
    test_size: float = 0.2,
    random_state: int = 42,
    confidence_level: float = 0.95,
) -> dict:
    """Fit an ordinary least-squares linear regression.

    Reports coefficients/intercept and a real confidence interval for each
    coefficient (standard OLS t-statistic, via the full dataset — the usual
    approach for inference on a small sample, where holding out a chunk
    just to fit would throw away scarce degrees of freedom for no benefit).
    R² is reported both on the full fit and on a held-out `test_size` split,
    since those answer different questions: r2_full is how well the line
    matches the data given to it, r2_test is a genuine (not
    training-set-inflated) check of whether the fit generalizes.
    """
    X_arr = _to_feature_matrix("X", X)
    y_arr = _to_array("y", y, min_dims=1, max_dims=1)
    if X_arr.shape[0] != y_arr.shape[0]:
        raise MLToolError(f"X has {X_arr.shape[0]} rows but y has {y_arr.shape[0]} values")
    _check_min_samples(X_arr.shape[0])
    if not (0.05 <= test_size <= 0.5):
        raise MLToolError("test_size must be between 0.05 and 0.5")
    if not (0.5 <= confidence_level < 1.0):
        raise MLToolError("confidence_level must be between 0.5 and 1.0 (exclusive)")

    n, p = X_arr.shape
    full_model = LinearRegression()
    full_model.fit(X_arr, y_arr)
    r2_full = r2_score(y_arr, full_model.predict(X_arr))

    dof = n - p - 1
    coefficient_standard_error: list[float] | None
    ci_lower: list[float] | None
    ci_upper: list[float] | None
    if dof >= 1:
        residuals = y_arr - full_model.predict(X_arr)
        residual_variance = float(np.sum(residuals**2) / dof)
        design = np.hstack([np.ones((n, 1)), X_arr])
        try:
            cov_matrix = residual_variance * np.linalg.inv(design.T @ design)
            se = np.sqrt(np.diag(cov_matrix))
            coefficient_standard_error = se[1:].tolist()
            t_crit = float(scipy.stats.t.ppf(1 - (1 - confidence_level) / 2, dof))
            ci_lower = [float(c - t_crit * s) for c, s in zip(full_model.coef_, se[1:])]
            ci_upper = [float(c + t_crit * s) for c, s in zip(full_model.coef_, se[1:])]
        except np.linalg.LinAlgError:
            coefficient_standard_error = ci_lower = ci_upper = None
    else:
        coefficient_standard_error = ci_lower = ci_upper = None

    X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=test_size, random_state=random_state)
    split_model = LinearRegression()
    split_model.fit(X_train, y_train)
    r2_train = r2_score(y_train, split_model.predict(X_train))
    r2_test = r2_score(y_test, split_model.predict(X_test))

    sanity_ok = -10.0 <= r2_test <= 1.0 + 1e-9
    sanity_note = "" if sanity_ok else f"r2_test={r2_test:.3f} is outside a plausible range — check the data/features"

    return {
        "coefficients": full_model.coef_.tolist(),
        "intercept": float(full_model.intercept_),
        "coefficient_standard_error": coefficient_standard_error,
        "confidence_level": confidence_level,
        "coefficient_ci_lower": ci_lower,
        "coefficient_ci_upper": ci_upper,
        "r2_full": float(r2_full),
        "r2_train": float(r2_train),
        "r2_test": float(r2_test),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_total": n,
        "sanity_ok": sanity_ok,
        "sanity_note": sanity_note,
    }


def fit_logistic_regression(X: list[list[float]], y: list[float], test_size: float = 0.2, random_state: int = 42) -> dict:
    """Fit a logistic regression classifier, holding out `test_size` of the data to report genuine test accuracy against a majority-class baseline."""
    X_arr = _to_feature_matrix("X", X)
    y_arr = _to_array("y", y, min_dims=1, max_dims=1)
    if X_arr.shape[0] != y_arr.shape[0]:
        raise MLToolError(f"X has {X_arr.shape[0]} rows but y has {y_arr.shape[0]} values")
    _check_min_samples(X_arr.shape[0])
    n_classes = len(np.unique(y_arr))
    if n_classes < 2:
        raise MLToolError("y must contain at least 2 distinct classes")
    if not (0.05 <= test_size <= 0.5):
        raise MLToolError("test_size must be between 0.05 and 0.5")

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=test_size, random_state=random_state, stratify=y_arr if n_classes <= len(y_arr) // 2 else None
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, model.predict(X_train))
    acc_test = accuracy_score(y_test, model.predict(X_test))

    values, counts = np.unique(y_test, return_counts=True)
    baseline = float(counts.max() / counts.sum())

    sanity_ok = 0.0 <= acc_test <= 1.0
    sanity_note = "" if sanity_ok else f"acc_test={acc_test:.3f} is outside [0, 1] — this should be impossible, check inputs"

    return {
        "accuracy_train": float(acc_train),
        "accuracy_test": float(acc_test),
        "baseline_accuracy": baseline,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "sanity_ok": sanity_ok,
        "sanity_note": sanity_note,
    }


def kmeans_cluster(X: list[list[float]], n_clusters: int, random_state: int = 42) -> dict:
    """Cluster rows into `n_clusters` groups with k-means, reporting a silhouette score (-1 to 1; higher means better-separated clusters) instead of just returning labels with no quality signal."""
    X_arr = _to_feature_matrix("X", X)
    n_samples = X_arr.shape[0]
    _check_min_samples(n_samples)
    if not (2 <= n_clusters <= n_samples - 1):
        raise MLToolError(f"n_clusters must be between 2 and {n_samples - 1} for {n_samples} samples, got {n_clusters}")

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_arr)
    score = silhouette_score(X_arr, labels)

    sanity_ok = -1.0 <= score <= 1.0
    sanity_note = "" if sanity_ok else f"silhouette_score={score:.3f} is outside [-1, 1] — this should be impossible, check inputs"

    return {
        "labels": labels.tolist(),
        "cluster_centers": model.cluster_centers_.tolist(),
        "silhouette_score": float(score),
        "sanity_ok": sanity_ok,
        "sanity_note": sanity_note,
    }


def pca_reduce(X: list[list[float]], n_components: int) -> dict:
    """Reduce X to `n_components` principal components, reporting how much variance is actually retained rather than assuming the reduction is lossless."""
    X_arr = _to_feature_matrix("X", X)
    n_samples, n_features = X_arr.shape
    _check_min_samples(n_samples)
    if not (1 <= n_components <= min(n_samples, n_features)):
        raise MLToolError(f"n_components must be between 1 and {min(n_samples, n_features)}, got {n_components}")

    model = PCA(n_components=n_components)
    transformed = model.fit_transform(X_arr)
    cumulative = float(np.sum(model.explained_variance_ratio_))

    sanity_ok = 0.0 <= cumulative <= 1.0 + 1e-9
    sanity_note = "" if sanity_ok else f"cumulative_explained_variance={cumulative:.3f} is outside [0, 1] — this should be impossible, check inputs"

    return {
        "transformed": transformed.tolist(),
        "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance": cumulative,
        "sanity_ok": sanity_ok,
        "sanity_note": sanity_note,
    }


TOOLS = {
    "fit_linear_regression": fit_linear_regression,
    "fit_logistic_regression": fit_logistic_regression,
    "kmeans_cluster": kmeans_cluster,
    "pca_reduce": pca_reduce,
}


def run_ml_tool(tool: str, args: dict) -> dict:
    """Dispatches by name — the single entry point POST /tasks/ml and
    app.agent's ml tool wrappers both call, so there's one place that
    validates the tool name itself."""
    fn = TOOLS.get(tool)
    if fn is None:
        raise MLToolError(f"Unknown ML tool '{tool}'. Expected one of: {sorted(TOOLS)}")
    return fn(**args)
