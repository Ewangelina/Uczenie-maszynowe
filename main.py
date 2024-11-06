from data_processing import load_and_process_movies, load_train_data, load_task_data
from train import find_best_k
from predict import predict_ratings, save_predictions

print("Step 1: Loading and processing movie data...")
movie_df = load_and_process_movies()

print("Step 2: Loading training data...")
X_train, y_train = load_train_data(movie_df)

# print("Step 3: Finding the best k with cross-validation...")
# best_k = find_best_k(X_train, y_train)
best_k = 20
# print(f"Optimal k found: {best_k}")

print("Step 4: Loading task data...")
task_df, X_task = load_task_data(movie_df)

print("Step 5: Predicting ratings...")
task_df = predict_ratings(X_train, y_train, X_task, best_k, task_df)

print("Step 6: Saving predictions to file...")
save_predictions(task_df)
print("Predictions saved to 'predicted_ratings.csv'")

