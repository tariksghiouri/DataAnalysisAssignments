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

    eigenvalues, eigenvectors = qr_algorithm(cov)

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

    projected_matrix = np.dot(centered_matrix, eigenvectors)

    return cov, eigenvalues, eigenvectors

def qr_algorithm(matrix, num_iterations=100):
    n = matrix.shape[0]
    A = matrix
    eigenvectors = np.identity(n)
    for _ in range(num_iterations):
        Q, R = np.linalg.qr(A)
        A = np.dot(R, Q)
        eigenvectors = np.dot(eigenvectors, Q)
    eigenvalues = np.diagonal(A)
    return eigenvalues, eigenvectors

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

# Scatter plot of the data points
plt.figure(figsize=(6, 6))
plt.scatter(eigenvectors[:, 0], eigenvectors[:, 1], c='b', marker='o', alpha=0.6)

# Calculate eigenvalue magnitudes for PCA circle
eigenvalue_magnitudes = np.sqrt(eigenvalues)

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

plt.title('PCA Circle (QR Algorithm)')
plt.grid()
plt.show()