import math
import numpy as np
import matplotlib.pyplot as plt

def display_matrix(matrix):
    max_width = max(len(str(element)) for row in matrix for element in row)

    for row in matrix:
        formatted_row = [str(element).rjust(max_width) for element in row]
        print(' '.join(formatted_row))
        
        
def calculate_mean(data):
    # Calculate the mean along each column (feature)
    num_samples = len(data)
    num_features = len(data[0])

    mean = [0] * num_features

    for sample in data:
        for i in range(num_features):
            mean[i] += sample[i]

    mean = [x / num_samples for x in mean]

    return mean


def calculate_covariance(data):
    # Calculate the covariance matrix
    num_samples = len(data)
    num_features = len(data[0])

    mean = calculate_mean(data)

    covariance_matrix = [[0] * num_features for _ in range(num_features)]

    for sample in data:
        for i in range(num_features):
            for j in range(num_features):
                covariance_matrix[i][j] += (sample[i] -
                                            mean[i]) * (sample[j] - mean[j])

    covariance_matrix = [[x / (num_samples - 1) for x in row]
                         for row in covariance_matrix]

    return covariance_matrix


def calculate_standard_deviation(data):
    num_dimensions = len(data[0])
    num_data_points = len(data)
    standard_deviations = []

    for dimension in range(num_dimensions):
        dimension_data = [point[dimension] for point in data]
        mean = sum(dimension_data) / num_data_points
        squared_diff_sum = sum((x - mean) ** 2 for x in dimension_data)
        variance = squared_diff_sum / num_data_points
        standard_deviation = math.sqrt(variance)
        standard_deviations.append(standard_deviation)

    return standard_deviations

data = [[1, 5, 3, 1],
        [4, 2, 6, 3],
        [1, 4, 3, 2],
        [4, 4, 1, 1],
        [5, 5, 2, 3]]
data2 = [[1, 2, 3, 4],
        [5, 5, 6, 7],
        [1, 4, 2, 3],
        [5, 3, 2, 1],   
        [8, 1, 2, 2]]


print(calculate_mean(data2))
print(calculate_standard_deviation(data))

def normalize_data(data):
    mean = calculate_mean(data)
    standard_deviations = calculate_standard_deviation(data)

    normalized_data = []

    for sample in data:
        normalized_sample = [(sample[i] - mean[i]) / standard_deviations[i] for i in range(len(sample))]
        normalized_data.append(normalized_sample)

    return normalized_data

normalized_data = normalize_data(data2)

for sample in normalized_data:
    formatted_sample = [f"{x:.2f}" for x in sample]
    print(formatted_sample)

covariance_matrix = calculate_covariance(normalized_data)




for row in covariance_matrix:
    formatted_row = [f"{x:.2f}" for x in row]
    print(formatted_row)
    
    
    
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvalues = eigenvalues[sorted_indices]
sorted_eigenvectors = eigenvectors[:, sorted_indices]


explained_variance_ratio = sorted_eigenvalues / np.sum(sorted_eigenvalues)

principal_components = sorted_eigenvectors

# Print the results
print("Eigenvalues:")
print(sorted_eigenvalues)
print("Explained Variance Ratio:")
print(explained_variance_ratio)
print("Principal Components:")
print(principal_components)

def reduce_dimensionality(data, num_components):
    normalized_data = normalize_data(data)
    covariance_matrix = calculate_covariance(normalized_data)

    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    principal_components = np.array(sorted_eigenvectors.T[:num_components, :])

    reduced_data = np.dot(normalized_data, principal_components.T)

    return reduced_data
reduced_data = reduce_dimensionality(data2, 2)
print(reduced_data)

