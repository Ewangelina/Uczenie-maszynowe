import json
import math

from sklearn.metrics import mean_squared_error
from data_processing import load_reviews, split_reviews
from similarity import create_similarity_list, predict

def grid_search(reviews, validation_data, thresholds):
    best_threshold = None
    best_rmse = float('inf')

    for threshold in thresholds:
        total_error = 0
        count = 0

        for row in validation_data:
            student_id, ratings = row[0], row[1:]
            for movie_id, actual_grade in enumerate(ratings):
                if actual_grade != 'X':
                    similarity_list = create_similarity_list(student_id, reviews, threshold)
                    predicted_grade = predict(movie_id, similarity_list)
                    if predicted_grade is not None:
                        total_error += (float(predicted_grade) - float(actual_grade)) ** 2
                        count += 1

        rmse = math.sqrt(total_error / count) if count > 0 else float('inf')
        print(f"Threshold: {threshold}, RMSE: {rmse}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_threshold = threshold

    return best_threshold, best_rmse

def save_hyperparameters(file_path, hyperparameters):
    with open(file_path, 'w') as f:
        json.dump(hyperparameters, f)

def load_hyperparameters(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

if __name__ == "__main__":
    reviews, _ = load_reviews()
    training_set, validation_set = split_reviews(reviews)
    thresholds = range(1, 91, 10)

    best_threshold, best_rmse = grid_search(training_set, validation_set, thresholds)
    print(f"Best Threshold: {best_threshold}, Best RMSE: {best_rmse}")

    save_hyperparameters("best_hyperparameters.json", {"no_movies_treshold": best_threshold})
