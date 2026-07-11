import numpy as np
from itertools import combinations

## ---- SECTION 1: THE SHORTEST PATH SETUP ---- ##

def create_paths() -> list:
    """
    Enumerates all 70 unique paths in the SPP

    Returns:
        paths (list): a list of strings where each string is a unique path
    """
    paths = []
    for positions in combinations(range(8), 4):
        path = ['R'] * 8
        for pos in positions:
            path[pos] = 'D'
        paths.append(''.join(path))
    return paths

def compute_paths(paths: list, D: int) -> np.ndarray:
    """
    Create the path-edge incidence matrix using the paths list

    Inputs:
        paths (list): a list of strings where each string is a unique path
        D (int): the number of edges in the SPP

    Returns:
        path_matrix (np.ndarray): a numpy array where each row is the incidence vector of a path on the edges
    """
    path_matrix = np.zeros((len(paths), D))
    for idx, path in enumerate(paths):
        row, col = 0, 0
        for move in path:
            if move == 'R':
                path_matrix[idx, row * 4 + col] = 1
                col += 1
            elif move == 'D':
                path_matrix[idx, 20 + row * 5 + col] = 1
                row += 1
    return path_matrix

def solve_shortest_path(path_matrix: np.ndarray, edge_costs: np.ndarray) -> np.ndarray:
    """
    Identifies which path is the shortest under the current cost vector

    Inputs:
        path_matrix (np.ndarray): The path-edge incidence matrix of the SPP
        edge_costs (np.ndarray): The cost vector of the SPP

    Returns:
        shortest_path (np.ndarray): The shortest path expressed as a binary vector 
    """
    costs = path_matrix @ edge_costs
    shortest_path = path_matrix[np.argmin(costs)]
    return shortest_path

def evaluate_path_cost(path_vector: np.ndarray, edge_costs: np.ndarray) -> float:
    """
    Evaluates the cost of a given path

    Inputs:
        path_vector (np.ndarray): The binary incidence vector for a path
        edge_costs (np.ndarray): The cost vector of the SPP
    
    Returns:
        path_cost (float): A float representing the cost of the path
    """
    path_cost = np.dot(path_vector, edge_costs)
    return path_cost

def generate_costs(X_matrix: np.ndarray, deg: int, h: float, D: int, P: int, B: np.ndarray, rng: np.random.default_rng) -> np.ndarray:
    """
    Generates costs of edges given a matrix of covariates X

    Input:
        X_matrix (np.ndarray): A matrix of covariates
        deg (int): The polynomial degree for the cost function
        h (float): The half width noise for the cost vector
        D (int): The number of edges
        P (int): The number of covariates
        B (np.ndarray): The ground truth matrix

    Returns:
        Y_matrix (np.ndarray): A matrix of costs for each sample/covariate
    """
    n = X_matrix.shape[0]
    Y_matrix = np.zeros((n, D))
    for t in range(n):
        y = (((1 / np.sqrt(P)) * np.dot(B, X_matrix[t]) + 3) ** deg + 1)
        if h > 0:
            y = y * rng.uniform(1 - h, 1 + h, size=D)
        Y_matrix[t] = y
    return Y_matrix

## ---- SECTION 2: THE DECISION MAKING SETUP ---- ##

def generate_test_val_train_data(D, P, B, deg, h, num_train, num_val, num_test, rng):
    """
    Generate testing, validation, and training data at once
    """
    
    X_train = rng.standard_normal(size=(num_train, P))
    Y_train = generate_costs(X_train, deg, h, D, P, B, rng)

    X_val = rng.standard_normal(size=(num_val, P))
    Y_val = generate_costs(X_val, deg, h, D, P, B, rng)

    X_test = rng.standard_normal(size=(num_test, P))
    Y_test = generate_costs(X_test, deg, h, D, P, B, rng)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

