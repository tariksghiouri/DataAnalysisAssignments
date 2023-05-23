import numpy as np
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
                covariance_matrix[i][j] += (sample[i] - mean[i]) * (sample[j] - mean[j])
    
    covariance_matrix = [[x / (num_samples - 1) for x in row] for row in covariance_matrix]
    
    return covariance_matrix
