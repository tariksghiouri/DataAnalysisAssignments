import numpy as np
def pca(X):
    # Compute the mean of the data
    mean_X = np.mean(X, axis=0)

    # Center the data
    X_centered = X - mean_X

    # Compute the covariance matrix
    cov_X = np.cov(X_centered, rowvar=False)

    # Compute the eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_X)

    # Sort the eigenvectors by eigenvalue in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Select the principal components
    explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
    cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative_explained_variance_ratio >= 0.95) + 1
    principal_components = eigenvectors[:, :n_components]

    # Transform the data
    X_pca = np.dot(X_centered, principal_components)
    return X_pca


X = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15]
])
X_pca = pca(X)
print(X_pca)