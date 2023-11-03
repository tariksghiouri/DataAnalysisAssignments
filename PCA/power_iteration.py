import numpy as np

def power_iteration(matrix, num_iterations):
    n = matrix.shape[0]
    
    # Initial guess 
    b = np.random.rand(n)
    
    for _ in range(num_iterations):
        # matrix-vector multiplication
        Ab = matrix @ b
        
        # Normalize 
        b = Ab / np.linalg.norm(Ab)
        
        # eigenvalue approximation
        eigenvalue = np.dot(b, matrix @ b)
    
    return eigenvalue, b

# Exemple d'utilisation
if __name__ == "__main__":
    A = np.array([[4, -2],
                  [1,  1]])
    
    num_iterations = 100  # itérations pour la méthode de puissance itérée
    
    eigenvalue, eigenvector = power_iteration(A, num_iterations)
    
    print("Valeur propre dominante approximee:", eigenvalue)
    print("Vecteur propre correspondant:", eigenvector)
