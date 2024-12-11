import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Step 1: Load and preprocess data
def load_data(data_dir):
    all_files = glob.glob(f"{data_dir}/*.csv")
    data = pd.concat([pd.read_csv(f, sep=';', names=['Row ID', 'User ID', 'Movie ID', 'Rating']) for f in all_files])
    return data

def create_rating_matrix(data):
    users = data['User ID'].unique()
    movies = data['Movie ID'].unique()
    user_map = {u: idx for idx, u in enumerate(users)}
    movie_map = {m: idx for idx, m in enumerate(movies)}

    R = np.zeros((len(users), len(movies)))
    for _, row in data.iterrows():
        R[user_map[row['User ID']], movie_map[row['Movie ID']]] = row['Rating']
    return R, user_map, movie_map

# Step 2: Matrix Factorization
class MatrixFactorization:
    def __init__(self, R, K, alpha, beta, iterations):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def train(self):
        self.U = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.V = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.R[self.R > 0])

        samples = [(i, j, self.R[i, j]) for i in range(self.num_users)
                   for j in range(self.num_items) if self.R[i, j] > 0]

        for _ in range(self.iterations):
            np.random.shuffle(samples)
            for i, j, r in samples:
                prediction = self.predict(i, j)
                e = r - prediction

                # Update biases
                self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
                self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

                # Update latent factors
                self.U[i, :] += self.alpha * (e * self.V[j, :] - self.beta * self.U[i, :])
                self.V[j, :] += self.alpha * (e * self.U[i, :] - self.beta * self.V[j, :])

    def predict(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.U[i, :].dot(self.V[j, :].T)
        return prediction

    def full_matrix(self):
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.U.dot(self.V.T)

# Step 3: Evaluate Model
def evaluate_model(R_test, R_pred):
    mask = R_test > 0
    mse = np.mean((R_test[mask] - R_pred[mask]) ** 2)
    return mse

# Step 4: Cross-Validation for Hyperparameter Tuning
def cross_validate(R, param_grid, k_folds=5):
    kf = KFold(n_splits=k_folds)
    best_params = None
    best_mse = float('inf')

    for K in param_grid['K']:
        for alpha in param_grid['alpha']:
            for beta in param_grid['beta']:
                print(f"Testing K={K}, alpha={alpha}, beta={beta}")
                mse_scores = []

                for train_idx, test_idx in kf.split(np.arange(R.shape[0])):
                    # Create train/test masks
                    train_mask = np.zeros_like(R, dtype=bool)
                    test_mask = np.zeros_like(R, dtype=bool)
                    train_mask[train_idx, :] = R[train_idx, :] > 0
                    test_mask[test_idx, :] = R[test_idx, :] > 0

                    # Train-test split
                    R_train = np.where(train_mask, R, 0)
                    R_test = np.where(test_mask, R, 0)

                    # Train model
                    mf = MatrixFactorization(R_train, K=K, alpha=alpha, beta=beta, iterations=50)
                    mf.train()

                    # Predict and evaluate
                    R_pred = mf.full_matrix()
                    R_pred = np.clip(np.round(R_pred), 0, 5)
                    mse = evaluate_model(R_test, R_pred)
                    mse_scores.append(mse)

                avg_mse = np.mean(mse_scores)
                print(f"Average MSE: {avg_mse}")

                if avg_mse < best_mse:
                    best_mse = avg_mse
                    best_params = {'K': K, 'alpha': alpha, 'beta': beta}

    return best_params, best_mse

# Step 5: Predict Task Data
def predict_task(task_data, R_pred, user_map, movie_map):
    predictions = []
    for _, row in task_data.iterrows():
        user_idx = user_map.get(row['User ID'], None)
        movie_idx = movie_map.get(row['Movie ID'], None)
        if user_idx is not None and movie_idx is not None:
            predictions.append(R_pred[user_idx, movie_idx])
        else:
            predictions.append(np.nan)  # Handle cold-start issues
    return predictions

# Usage
data_dir = 'Dane/train'
task_file = 'collaborative-filtering/task.csv'
data = load_data(data_dir)

# Create user-item matrix
R, user_map, movie_map = create_rating_matrix(data)

# Hyperparameter grid
param_grid = {
    'K': [10, 20, 30],
    'alpha': [0.001, 0.01, 0.1],
    'beta': [0.001, 0.01, 0.1]
}

# Cross-validate to find best parameters
best_params, best_mse = cross_validate(R, param_grid, k_folds=5)
print("Best Parameters:", best_params)
print("Best MSE:", best_mse)

# Train final model with best parameters
mf = MatrixFactorization(R, K=best_params['K'], alpha=best_params['alpha'], beta=best_params['beta'], iterations=100)
mf.train()

# Predict task data
R_pred = mf.full_matrix()
task_data = pd.read_csv(task_file, sep=';', names=['Row ID', 'User ID', 'Movie ID', 'Rating'])
R_pred_clipped = np.clip(np.round(R_pred), 0, 5)
task_data['Predicted Rating'] = predict_task(task_data, R_pred_clipped, user_map, movie_map)

# Save predictions
task_data.to_csv('predictions.csv', sep=';', index=False)
