import random

random.seed(42)

def load_reviews(file_path="./Dane/student_reviews.csv"):
    with open(file_path, "r") as f:
        result = []
        idiots = []
        for line in f:
            split_line = line.strip().split(";")
            if is_user_stupid(split_line):
                idiots.append(split_line)
            else:
                result.append(split_line)
    return result, idiots

def is_user_stupid(row):
    usual_grade = row[1]
    for i in range(2, len(row)):
        if row[i] == usual_grade:
            return False
    return True

def split_reviews(reviews, validation_ratio=0.2):
    training_set = []
    validation_set = []

    for review in reviews:
        student_id = review[0]
        ratings = review[1:]  # Exclude the student ID

        rated_movies = [(i, float(rating)) for i, rating in enumerate(ratings) if rating != 'X']
        num_validation = int(len(rated_movies) * validation_ratio)

        validation_indices = random.sample(rated_movies, num_validation)

        validation_row = ['X'] * len(ratings)
        for idx, rating in validation_indices:
            validation_row[idx] = rating

        training_row = ratings[:]
        for idx, _ in validation_indices:
            training_row[idx] = 'X'

        training_set.append([student_id] + training_row)
        validation_set.append([student_id] + validation_row)

    return training_set, validation_set
