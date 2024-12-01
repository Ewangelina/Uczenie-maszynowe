import csv
from data_processing import load_reviews
from similarity import create_similarity_list, predict
from hyperparameter_tuning import load_hyperparameters

reviews, _ = load_reviews()
hyperparameters = load_hyperparameters("best_hyperparameters.json")
best_threshold = hyperparameters["no_movies_treshold"] if hyperparameters else 1

with open("task.csv", "r") as task_file, open("output.csv", "w", newline='') as output_file:
    reader = csv.reader(task_file, delimiter=';')
    writer = csv.writer(output_file, delimiter=';')

    current_student_id = None
    similarity_list = None

    for row in reader:
        task_id, student_id, movie_id, grade = row

        if grade == "NULL":
            if student_id != current_student_id:
                current_student_id = student_id
                similarity_list = create_similarity_list(student_id, reviews, best_threshold)

            grade = predict(movie_id, similarity_list) or "NULL"

        writer.writerow([task_id, student_id, movie_id, grade])
