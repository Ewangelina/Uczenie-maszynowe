from data_processing import load_and_process_movies, load_train_data, load_task_data
from train import find_best_k
from predict import predict_ratings, save_predictions
from os import listdir
from os.path import isfile, join

all_train_files = listdir(".\\Dane\\train")
print("Preprocess: Loading and processing movie data...")
movie_df = load_and_process_movies()

for file in all_train_files:
    trainFilePath = ".\\Dane\\train\\" + file
    print(f"Loading training data for student {file}")
    X_train, y_train = load_train_data(movie_df, trainFilePath)

    best_k = find_best_k(X_train, y_train)
    print(f"Optimal k found: {best_k}")

    taskFilePath = ".\\Dane\\task\\" + file
    task_df, X_task = load_task_data(movie_df, taskFilePath)

    print(f"Predicting ratings for student {file}")
    task_df = predict_ratings(X_train, y_train, X_task, best_k, task_df)

    print(f"Saving predictions for {file}")
    resultFilePath = ".\\Dane\\output\\" + file
    save_predictions(task_df, resultFilePath)

print("TASK DONE")
