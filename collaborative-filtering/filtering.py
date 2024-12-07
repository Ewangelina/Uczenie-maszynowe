import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

# Step 3: Evaluate and predict
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

# Main script
if __name__ == "__main__":
    # File paths
    data_dir = 'Dane/train'
    task_file = 'collaborative-filtering/task.csv'
    
    # Step 1: Load and preprocess data
    data = load_data(data_dir)
    
    # Step 2: Create user-item matrix
    R, user_map, movie_map = create_rating_matrix(data)
    
    # Step 3: Train the model
    mf = MatrixFactorization(R, K=20, alpha=0.01, beta=0.01, iterations=100)
    mf.train()
    
    # Step 4: Predict task data
    task_data = pd.read_csv(task_file, sep=';', names=['Row ID', 'User ID', 'Movie ID', 'Rating'])
    R_pred = mf.full_matrix()
    
    # Round and clip predictions to valid range
    R_pred = np.clip(np.round(R_pred), 0, 5)
    
    # Generate predictions for the task data
    task_data['Predicted Rating'] = predict_task(task_data, R_pred, user_map, movie_map)
    
    # Step 5: Save predictions
    task_data.to_csv('predictions.csv', sep=';', index=False)
