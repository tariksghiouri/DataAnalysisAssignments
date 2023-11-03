import numpy as np
# import exceptions as ex


class PCA:

    def __init__(self, n_components=3):
        self.means_ = None
        self.stds_ = None
        self.cov_mat_ = None
        self.eigen_values_ = None
        self.eigen_vectors_ = None
        self.pcs = None
        self.n_principal_components = n_components  # Initialize n_principal_components here

    def standardizer(self, data: np.ndarray = None):
        """
        Standardizer is method to standardize a matrix.
        :param data: dataset (matrix)
        :return: standardized matrix.
        """
        means = np.zeros((data.shape[1]))
        standard_deviations = np.zeros((data.shape[1]))
        for i in range(data.shape[1]):
            means[i] = np.mean(data[:, i])
            standard_deviations[i] = np.std(data[:, i])

        standardized_data = np.ones(shape=data.shape)
        for i in range(standardized_data.shape[0]):
            for j in range(standardized_data.shape[1]):
                standardized_data[i][j] = (data[i][j] - means[j])/standard_deviations[j]

        self.means_ = means
        self.stds_ = standard_deviations
        return standardized_data

    def covariance_matrix(self, data):
        """
        covariance_matrix is a helper function used to compute the covariance matrix of a given matrix.
        :param data: dataset (matrix)
        :return: covariance matrix
        """
        cov_matrix = np.ones(shape=(data.shape[1], data.shape[1]))
        for i in range(cov_matrix.shape[0]):
            for j in range(cov_matrix.shape[0]):
                cov_matrix[i][j] = PCA.covariance(data[:, i], data[:, j])
        self.cov_mat_ = cov_matrix
        return cov_matrix

    @staticmethod
    def covariance(x: np.ndarray = None, y: np.ndarray = None):
        """
        covariance is used to compute the covariance of two vectors.
        :param x: the first vector
        :param y: the second vector
        :return: covariance of x and y.
        """
        return np.sum((x - np.mean(x))*(y - np.mean(y)))/len(x)

    def transform(self, data: np.ndarray = None):
        """
        transform is the method used to generate the principal components of the new projection matrix, over which the
        dataset will get projected.
        :param data: dataset (matrix)
        :return: new projected matrix.
        """
        if self.n_principal_components > data.shape[1]:
            raise ex.KOORange
        ###### DATA STANDARDIZATION ######
        std_data_matrix = self.standardizer(data)

        ###### COVARIANCE MATRIX ######
        cov_matrix = self.covariance_matrix(std_data_matrix)

        ###### EIGEN VECTORS & EIGEN VALUES ######
        eigens = np.linalg.eig(cov_matrix)
        eigen_vectors = eigens[1]
        signs = np.sign(eigen_vectors[np.argmax(np.abs(eigens[1]), axis=0), range(eigen_vectors.shape[0])])
        eigen_vectors = eigens[1] * signs[np.newaxis, :]
        eigen_vectors = eigen_vectors.T
        eigen_combo = [(eigens[0][i], eigen_vectors[i, :]) for i in range(len(eigens[0]))]

        eigen_combo.sort(key=lambda x: x[0], reverse=True)
        ###### EIGEN CONTRIBUTIONS ######
        self.eigen_values_ = np.array([x[0] for x in eigen_combo])
        self.eigen_vectors_ = np.array([x[1] for x in eigen_combo])

        projection_matrix = self.eigen_vectors_[:self.n_principal_components, :]

        ###### TRANSFORMED MATRIX ######
        np.set_printoptions(suppress=True)
        transformed_matrix = std_data_matrix.dot(projection_matrix.T)

        return transformed_matrix
    
    

import numpy as np

pca = PCA()

data_matrix = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

standardized_data = pca.standardizer(data_matrix)

transformed_matrix = pca.transform(standardized_data)

print(transformed_matrix)


