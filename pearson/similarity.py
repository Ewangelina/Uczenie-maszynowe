import random

def calculate_similarity(row1, row2, no_movies_treshold):
    #tu pearson
    #jeśli ilość filmów zrecenzowanych przez obie osoby < no_movies_treshold to zwróć None
    return random.random()
    
def predict(movie_id, similarity_list):
    i = 0
    movie_id = int(movie_id)
    for i in range(len(similarity_list)):
        if similarity_list[i][movie_id] == 'X':
            i = i + 1
        else:
            return similarity_list[i][movie_id]

    return None
        
def create_similarity_list(student_id, reviews, no_movies_treshold):
    current_student = None
    current_student_index = -1
    similarity_list = []
    
    for i in range(len(reviews)):
        if reviews[i][0] == student_id:
            current_student = reviews[i]
            current_student_index = i
            i = 999
            break
    #ewentualnie tu jeszcze obliczyć wszystkie wartości dla aktualnego studenta do pearsona
    for i in range(len(reviews)):
        if not i == current_student_index:
            distance = calculate_similarity(current_student, reviews[i], no_movies_treshold)
            if not distance is None:
                student_copy = reviews[i]
                student_copy[0] = distance
                similarity_list.append(student_copy)

    #tu posortuj similarity_list wg [0]
    return similarity_list
    
        
