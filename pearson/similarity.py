from data_processing import calculate_movie_average


def calculate_similarity(row1, row2, no_movies_treshold):
    common_movies = [
        (float(row1[i]), float(row2[i]))
        for i in range(1, len(row1))
        if row1[i] != 'X' and row2[i] != 'X'
    ]
    if len(common_movies) < no_movies_treshold:
        return None
    
    scores_row1, scores_row2 = zip(*common_movies)
    mean_row1, mean_row2 = sum(scores_row1)/len(scores_row1), sum(scores_row2)/len(scores_row2)
    numerator = sum((a - mean_row1) * (b - mean_row2) for a, b in common_movies)
    denominator = (
        (sum((a - mean_row1) ** 2 for a in scores_row1) ** 0.5) *
        (sum((b - mean_row2) ** 2 for b in scores_row2) ** 0.5)
    )
    similarity = numerator / denominator if denominator != 0 else 0
    return similarity


def predict(movie_id, similarity_list, reviews, global_average=2.5, min_ratings=3):
    movie_id = int(movie_id)
    weighted_sum = 0
    total_similarity = 0
    valid_ratings = 0

    # Calculate prediction using similarity scores
    for similarity, row in similarity_list:
        rating = row[movie_id]
        if rating != 'X':  # If the user has reviewed the movie
            weighted_sum += abs(similarity) * float(rating)
            total_similarity += abs(similarity)
            valid_ratings += 1

        if valid_ratings >= min_ratings:  # Stop if we have enough ratings
            break

    if total_similarity > 0:
        predicted_grade = weighted_sum / total_similarity
        predicted_grade = max(0, min(5, predicted_grade))  # Ensure valid range
        return round(predicted_grade)

    # Fallback: Movie average
    movie_average = calculate_movie_average(movie_id, reviews)
    if movie_average is not None:
        return round(movie_average)

    # Final fallback: Global average
    print(f"Using global average for movie {movie_id}: {global_average}")
    return round(global_average)


def create_similarity_list(student_id, reviews, no_movies_treshold):
    current_student = None
    similarity_list = []

    for row in reviews:
        if row[0] == student_id:
            current_student = row
            break

    if not current_student:
        print(f"Student {student_id} not found in reviews.")
        return similarity_list

    for row in reviews:
        if row[0] != student_id:
            similarity = calculate_similarity(current_student, row, no_movies_treshold)
            if similarity is not None:
                similarity_list.append((similarity, row))

    similarity_list.sort(reverse=True, key=lambda x: x[0])
    return similarity_list
