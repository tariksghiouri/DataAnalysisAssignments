import numpy as np


def jacobi_eigenvalue(A, tol=1e-10, max_iter=100):
    n = len(A)
    V = np.identity(n)  # Matrice de transformation initiale (matrice identité)
    num_iter = 0

    while num_iter < max_iter:
        off_diag = np.abs(A - np.diag(np.diag(A))).max()

        if off_diag < tol:
            break

        max_indices = np.unravel_index(
            np.argmax(np.abs(A - np.diag(np.diag(A))), axis=None), A.shape)
        i, j = max_indices
        theta = 0.5 * np.arctan2(2 * A[i, j], A[j, j] - A[i, i])

        R = np.identity(n)
        R[i, i] = np.cos(theta)
        R[j, j] = np.cos(theta)
        R[i, j] = -np.sin(theta)
        R[j, i] = np.sin(theta)

        A = np.dot(np.dot(R.T, A), R)
        V = np.dot(V, R)
        num_iter += 1

    eigenvalues = np.diag(A)
    eigenvectors = V
    return eigenvalues, eigenvectors


# Exemple d'utilisation
if __name__ == "__main__":
    # Remplacez cette matrice par votre matrice symétrique
    A = np.array([[4, -2],
              [1,  1]])

eigenvalues, eigenvectors = jacobi_eigenvalue(A)

print("Valeurs propres:", eigenvalues)
print("Vecteurs propres correspondants:\n", eigenvectors)
