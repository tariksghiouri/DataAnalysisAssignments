import numpy as np
import matplotlib.pyplot as plt

def pca(matrix):
    n_rows, n_columns = len(matrix), len(matrix[0])

    mean = [0] * n_columns
    for j in range(n_columns):
        column_sum = 0
        for i in range(n_rows):
            column_sum += matrix[i][j]
        mean[j] = column_sum / n_rows

    centered_matrix = [[matrix[i][j] - mean[j] for j in range(n_columns)] for i in range(n_rows)]

    cov = np.cov(np.array(centered_matrix), rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort eigenvalues and eigenvectors in descending order
    eigenvalue_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalue_indices]
    eigenvectors = eigenvectors[:, eigenvalue_indices]

    print("Eigenvalues:")
    for value in eigenvalues:
        print(value)

    total_variance = np.sum(eigenvalues)

    print("Variance Explained by Each Eigenvector:")
    for i, value in enumerate(eigenvalues):
        explained_variance = (value / total_variance) * 100
        print(f"Eigenvector {i + 1}: {explained_variance:.2f}%")

    print("Eigenvectors:")
    for row in eigenvectors.T:
        print(row)

    return cov, eigenvalues, eigenvectors

# Define Jacobi method for eigenvalue computation
def jacobi_eigenvalue(cov_matrix, tolerance=1e-6):
    n = cov_matrix.shape[0]
    eigenvalues = np.copy(np.diag(cov_matrix))

    max_iterations = 1000
    iteration = 0

    while iteration < max_iterations:
        max_off_diagonal = 0
        p, q = 0, 0

        # Find the maximum off-diagonal element
        for i in range(n - 1):
            for j in range(i + 1, n):
                if abs(cov_matrix[i, j]) > max_off_diagonal:
                    max_off_diagonal = abs(cov_matrix[i, j])
                    p, q = i, j

        if max_off_diagonal < tolerance:
            break

        # Compute Jacobi rotation angle (theta)
        a_ip = cov_matrix[p, p]
        a_iq = cov_matrix[p, q]
        a_qq = cov_matrix[q, q]
        if np.isclose(a_ip, a_qq):
            theta = np.pi / 4.0  # Set theta to 45 degrees if a_ip and a_qq are close
        else:
            theta = 0.5 * np.arctan(2 * a_iq / (a_ip - a_qq))

        # Compute Jacobi rotation matrix
        J = np.eye(n)
        J[p, p] = np.cos(theta)
        J[q, q] = np.cos(theta)
        J[p, q] = -np.sin(theta)
        J[q, p] = np.sin(theta)

        # Update the covariance matrix
        cov_matrix = np.dot(np.dot(J.T, cov_matrix), J)

        eigenvalues = np.copy(np.diag(cov_matrix))

        iteration += 1

    return eigenvalues

matrix = [[6, 2],
          [1, 6],
          [4, 5],
          [5, 3],
          [3, 4],
          [2, 1],
          [9, 8],
          [7, 9],
          [8, 7],
          [10, 10],
          [11, 11]]

cov_matrix, eigenvalues, eigenvectors = pca(matrix)

# Calculate eigenvalues using Jacobi method
eigenvalues_jacobi = jacobi_eigenvalue(cov_matrix)

# Scatter plot of the data points
plt.figure(figsize=(6, 6))
plt.scatter(eigenvectors[:, 0], eigenvectors[:, 1], c='b', marker='o', alpha=0.6)

# Calculate eigenvalue magnitudes for PCA circle
eigenvalue_magnitudes = np.sqrt(eigenvalues_jacobi)

# Draw PCA circle
for magnitude, vector in zip(eigenvalue_magnitudes, eigenvectors.T):
    plt.quiver(0, 0, vector[0], vector[1], angles='xy', scale_units='xy', scale=1, color='r', alpha=0.5)
    ellipse = plt.matplotlib.patches.Ellipse((0, 0), width=magnitude * 2, height=magnitude * 2, fill=False, color='r', alpha=0.5)
    plt.gca().add_patch(ellipse)

# Set axis limits
plt.xlim(-15, 15)
plt.ylim(-15, 15)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.title('PCA Circle (Jacobi Method)')
plt.grid()
plt.show()