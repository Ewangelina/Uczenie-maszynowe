import math

def calculate_similarity(row1, row2, no_movies_treshold):
    common_movies = [
        (float(row1[i]), float(row2[i]))
        for i in range(1, len(row1))
        if row1[i] != 'X' and row2[i] != 'X'
    ]
    if len(common_movies) < no_movies_treshold:
        return None
    
    # Pearson calculation (replace random placeholder)
    scores_row1, scores_row2 = zip(*common_movies)
    mean_row1, mean_row2 = sum(scores_row1)/len(scores_row1), sum(scores_row2)/len(scores_row2)
    numerator = sum((a - mean_row1) * (b - mean_row2) for a, b in common_movies)
    denominator = (
        (sum((a - mean_row1) ** 2 for a in scores_row1) ** 0.5) *
        (sum((b - mean_row2) ** 2 for b in scores_row2) ** 0.5)
    )
    similarity = numerator / denominator if denominator != 0 else 0
    return similarity


def predict(movie_id, similarity_list):
    movie_id = int(movie_id)
    weighted_sum = 0
    total_similarity = 0

    for similarity, row in similarity_list:
        rating = row[movie_id]
        if rating != 'X':  # If the user has reviewed the movie
            weighted_sum += similarity * float(rating)
            total_similarity += similarity

    # Return prediction if valid similarities exist
    return weighted_sum / total_similarity if total_similarity != 0 else None



def create_similarity_list(student_id, reviews, no_movies_treshold):
    current_student = None
    similarity_list = []

    for row in reviews:
        if row[0] == student_id:
            current_student = row
            break

    if not current_student:
        return similarity_list

    for row in reviews:
        if row[0] != student_id:
            similarity = calculate_similarity(current_student, row, no_movies_treshold)
            if similarity is not None:
                similarity_list.append((similarity, row))

    similarity_list.sort(reverse=True, key=lambda x: x[0])
    return similarity_list
