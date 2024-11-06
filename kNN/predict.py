import pandas as pd
from kNN import kNN

def predict_ratings(X_train, y_train, X_task, best_k, task_df):
    print("Training kNN model with best k:", best_k)
    knn = kNN(k=best_k)
    knn.fit(X_train, y_train)
    
    print("Predicting ratings for task data...")
    task_df["Predicted Rating"] = knn.predict(X_task)
    print("Ratings predicted successfully.")
    return task_df


def save_predictions(task_df, file_path="predicted_ratings.csv"):
    task_df[["User ID", "Predicted Rating"]].to_csv(file_path, index=False)
