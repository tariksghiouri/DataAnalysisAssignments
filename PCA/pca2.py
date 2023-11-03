import numpy as np
import matplotlib as plt

def pca(matrix):
    n_rows, n_columns = len(matrix), len(matrix[0])

    mean = [0] * n_columns
    for j in range(n_columns):
        column_sum = 0
        for i in range(n_rows):
            column_sum += matrix[i][j]
        mean[j] = column_sum / n_rows

    centered_matrix = [[matrix[i][j] - mean[j] for j in range(n_columns)] for i in range(n_rows)]
    print("Centered Matrix:")
    for row in centered_matrix:
        print(row)

    sigma = [0] * n_columns
    for j in range(n_columns):
        sum_squared = 0
        for i in range(n_rows):
            sum_squared += centered_matrix[i][j] ** 2
        sigma[j] = (sum_squared / n_rows) ** 0.5

    reduced_matrix = [[centered_matrix[i][j] / sigma[j] for j in range(n_columns)] for i in range(n_rows)]
    print("Centered and Reduced Matrix:")
    for row in reduced_matrix:
        print(row)

    cov = [[0] * n_columns for _ in range(n_columns)]
    for i in range(n_columns):
        for j in range(n_columns):
            e_sum = 0
            for k in range(n_rows):
                e_sum += centered_matrix[k][i] * centered_matrix[k][j]
            cov[i][j] = e_sum / n_rows
    print("Covariance Matrix:")
    for row in cov:
        print(row)

    corr = [[0] * n_columns for _ in range(n_columns)]
    for i in range(n_columns):
        for j in range(n_columns):
            corr[i][j] = cov[i][j] / (sigma[i] * sigma[j])
    print("Correlation Matrix:")
    for row in corr:
        print(row)

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

    projected_matrix = np.dot(centered_matrix, eigenvectors)

    return projected_matrix


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

projected_matrix = pca(matrix)

print("Projected Matrix:")
for row in projected_matrix:
    print(row)
# Calculate the correlations between the original variables and principal components
correlation_matrix = np.corrcoef(np.array(matrix).T, projected_matrix.T)

# Plot the PCA circle of correlation
fig, ax = plt.subplots()
for i, (ex, ey) in enumerate(zip(eigenvectors[0], eigenvectors[1])):
    ax.arrow(0, 0, ex, ey, head_width=0.1, head_length=0.1, fc=f'C{i}', ec=f'C{i}', label=f'PC{i+1}')

# Add variable names to the arrows
for i, var in enumerate(matrix):
    ax.text(eigenvectors[0, i], eigenvectors[1, i], var, ha='center', va='center')

# Set axis limits
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

# Add labels and legend
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(loc='upper left')

plt.grid()
plt.show()