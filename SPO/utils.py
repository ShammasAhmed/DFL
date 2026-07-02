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

def create_path_matrix(paths: list, D: int) -> np.ndarray:
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

def generate_costs(X_matrix: np.ndarray, deg: int, h: float, D: int, P: int, curr_B: np.ndarray, rng: np.random.default_rng) -> np.ndarray:
    """
    Generates costs of edges given a matrix of covariates X

    Input:
        X_matrix (np.ndarray): A matrix of covariates
        deg (int): The polynomial degree for the cost function
        h (float): The half width noise for the cost vector
        D (int): The number of edges
        P (int): The number of features
        curr_B (np.ndarray): A matrix 

    Returns:
        C_mat (np.ndarray): A matrix of costs for each sample/covariate
    """
    n = X_matrix.shape[0]
    C_matrix = np.zeros((n, D))
    for t in range(n):
        c = (((1 / np.sqrt(P)) * np.dot(curr_B, X_matrix[t]) + 3) ** deg + 1)
        if h > 0:
            c = c * rng.uniform(1 - h, 1 + h, size=D)
        C_matrix[t] = c
    return C_matrix

## ---- SECTION 2: THE DECISION MAKING SETUP ---- ##

def generate_test_val_train_data(D, P, curr_B, deg, H, num_train, num_val, num_test, rng):
    """
    Generate testing, validation, and training data at once
    """
    
    X_train = rng.standard_normal(size=(num_train, P))
    C_train = generate_costs(X_train, deg, H, D, P, curr_B, rng)

    X_val = rng.standard_normal(size=(num_val, P))
    C_val = generate_costs(X_val, deg, H, D, P, curr_B, rng)

    X_test = rng.standard_normal(size=(num_test, P))
    C_test = generate_costs(X_test, deg, H, D, P, curr_B, rng)

    return X_train, C_train, X_val, C_val, X_test, C_test

def model_spo_regret(model, path_matrix, X_val, C_val, num_val, scale_factor = 1):
    """
    Total validation SPO regret of a fitted sklearn regressor on scaled costs.

    Input:
        model: A fitted sklearn regressor mapping covariates to edge costs
        path_matrix (np.ndarray): The path-edge incidence matrix of the SPP
        X_val (np.ndarray): Validation covariates
        C_val (np.ndarray): Validation costs
        num_val (int): Validation set size
        scale_factor (float): Scales costs to [0, 1] to assist GBM

    Returns:
        regret (float): The total regret of the model over the validation set
    """
    reg = 0.0
    for v in range(num_val):
        c_v = C_val[v]
        z_v = evaluate_path_cost(solve_shortest_path(path_matrix, c_v), c_v)
        p_v = np.clip(model.predict(X_val[v].reshape(1, -1)).flatten() * scale_factor, 0.001, None)
        reg += evaluate_path_cost(solve_shortest_path(path_matrix, p_v), c_v) - z_v
    return reg

