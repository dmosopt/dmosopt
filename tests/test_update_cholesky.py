"""
Analysis of CMAES updateCholesky Implementation
"""

import numpy as np


def updateCholesky(A, Ainv, z, psucc, pc, cc, ccov, pthresh):
    """

    Implements rank-1 Cholesky update for covariance matrix:
    C_new = alpha * C + beta * v .o v^T

    where C = A @ A.T and v = pc (evolution path).

    The update maintains:
    - A is lower Cholesky factor: C = A @ A.T
    - Ainv = A^(-1)

    Args:
        A: Lower Cholesky factor (n, n)
        Ainv: Inverse of A (n, n)
        z: Normalized step (n,)
        psucc: Success probability scalar
        pc: Evolution path (n,)
        cc: Cumulation time horizon
        ccov: Covariance learning rate
        pthresh: Threshold success rate

    Returns:
        Updated A, Ainv, pc
    """
    alpha = None
    if psucc < pthresh:
        # Active cumulation: successful step
        pc = (1.0 - cc) * pc + np.sqrt(cc * (2.0 - cc)) * z
        alpha = 1.0 - ccov
    else:
        # Passive cumulation: unsuccessful step
        pc = (1.0 - cc) * pc
        alpha = (1.0 - ccov) + ccov * cc * (2.0 - cc)

    beta = ccov
    w = np.dot(Ainv, pc)  # Normalized direction: A^(-1) @ pc

    # Only update if w is numerically significant
    if w.max() > 1e-20:
        w_times_Ainv = np.dot(w, Ainv)  # FIX 1: Use this variable

        # Scaling factors for rank-1 update
        a = np.sqrt(alpha)
        norm_w2 = np.sum(w**2)
        root = np.sqrt(1 + beta / alpha * norm_w2)
        b = a / norm_w2 * (root - 1)

        # Update A: Cholesky factor
        # Formula: A_new = a * A + b * (pc ⊗ w^T)
        # Note: pc = A @ w, so this is equivalent to standard form
        A = a * A + b * np.outer(pc, w)  # FIX 3: Removed unnecessary .T

        # Update Ainv: Inverse Cholesky factor
        # Formula: Ainv_new = (1/a) * Ainv - c * (w ⊗ (w @ Ainv)^T)
        c = 1.0 / (a * norm_w2) * (1.0 - 1.0 / root)
        Ainv = (1.0 / a) * Ainv - c * np.outer(w, w_times_Ainv)  # FIX 2

    return A, Ainv, pc


def test_updateCholesky():
    """Verify updateCholesky maintains proper matrix relationships."""
    np.random.seed(42)
    n = 5

    # Create valid initial Cholesky decomposition
    C = np.eye(n) + 0.1 * np.random.randn(n, n)
    C = C @ C.T  # Make positive definite
    A = np.linalg.cholesky(C)
    Ainv = np.linalg.inv(A)

    # Random evolution path and parameters
    pc = np.random.randn(n)
    z = np.random.randn(n)
    psucc = 0.3
    cc, ccov, pthresh = 0.2, 0.1, 0.44

    A_fix, Ainv_fix, pc_fix = updateCholesky(
        A.copy(), Ainv.copy(), z.copy(), psucc, pc.copy(), cc, ccov, pthresh
    )

    # A @ Ainv should be identity
    print("Test 1: A @ Ainv = I")
    print(f"  Max error:    {np.max(np.abs(A_fix @ Ainv_fix - np.eye(n))):.6e}")

    # Covariance matrix should be symmetric
    print("\nTest 2: C = A @ A.T is symmetric")
    C_fix = A_fix @ A_fix.T
    print(f"  Symmetry error:    {np.max(np.abs(C_fix - C_fix.T)):.6e}")

    # Covariance should be positive definite
    print("\nTest 3: C is positive definite")
    eig_fix = np.linalg.eigvalsh(C_fix)
    print(f"  Min eigenvalue:    {eig_fix.min():.6e}")


if __name__ == "__main__":
    test_updateCholesky()
