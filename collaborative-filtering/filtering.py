import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

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
        self.b = np.nanmean(self.R[self.R > 0])

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
        return np.clip(prediction, 0, 5)  # Ensure valid range

    def full_matrix(self):
        R_pred = self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.U.dot(self.V.T)
        return np.clip(R_pred, 0, 5)  # Ensure valid range

# Step 3: Cross-validation for hyperparameter tuning
def cross_validate(data, K_values, alpha_values, beta_values, iterations, n_splits=5):
    R, user_map, movie_map = create_rating_matrix(data)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_rmse = float('inf')
    best_params = None

    for K in K_values:
        for alpha in alpha_values:
            for beta in beta_values:
                rmse_folds = []

                for train_idx, test_idx in kf.split(R):
                    R_train = np.copy(R)
                    R_test = np.copy(R)

                    R_train[test_idx, :] = 0
                    R_test[train_idx, :] = 0

                    mf = MatrixFactorization(R_train, K, alpha, beta, iterations)
                    mf.train()

                    R_pred = mf.full_matrix()

                    # Validate predictions
                    valid_mask = ~np.isnan(R_test) & (R_test > 0)
                    if valid_mask.sum() > 0:
                        rmse = np.sqrt(mean_squared_error(R_test[valid_mask], R_pred[valid_mask]))
                        rmse_folds.append(rmse)

                avg_rmse = np.mean(rmse_folds)
                if avg_rmse < best_rmse:
                    best_rmse = avg_rmse
                    best_params = {'K': K, 'alpha': alpha, 'beta': beta}

    return best_params, best_rmse

# Main script
data_dir = 'Dane/train'
data = load_data(data_dir)

# Define hyperparameters to test
K_values = [10, 20, 30]
alpha_values = [0.01, 0.02, 0.05]
beta_values = [0.01, 0.02, 0.05]
iterations = 50

# Find best parameters
best_params, best_rmse = cross_validate(data, K_values, alpha_values, beta_values, iterations)
print("Best Parameters:", best_params)
print("Best RMSE:", best_rmse)

# Train with best parameters and predict task data
R, user_map, movie_map = create_rating_matrix(data)
mf = MatrixFactorization(R, best_params['K'], best_params['alpha'], best_params['beta'], iterations)
mf.train()

R_pred = mf.full_matrix()
task_file = 'collaborative-filtering/task.csv'
task_data = pd.read_csv(task_file, sep=';', names=['Row ID', 'User ID', 'Movie ID', 'Rating'])

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

task_data['Predicted Rating'] = predict_task(task_data, R_pred, user_map, movie_map)
task_data['Predicted Rating'] = task_data['Predicted Rating'].round().astype(int)

# Save predictions
task_data.to_csv('predictions.csv', sep=';', index=False)
