from data_processing import load_reviews
from similarity import predict, create_similarity_list

def write(string, file):
    o = open(file, "a")
    o.write(string)
    o.close()
    
reviews, idiots = load_reviews()
skip_list = []

for el in idiots:
    skip_list.append(el[0])

task = open(".\\..\\Dane\\task.csv")
current_student_id = -1
similarity_list = None
out_line = ""

for task_line in task:
    task_id, student_id, movie_id, grade = task_line.split(";")
    if student_id in skip_list:
        for el in idiots:
            if idiots[0] == student_id:
                grade = idiots[1]
                out_line = task_id + ";" + student_id + ";" + movie_id + ";" + grade + "\n"
    else:
        if student_id == current_student_id:
            grade = predict(movie_id, similarity_list)
            out_line = task_id + ";" + student_id + ";" + movie_id + ";" + grade + "\n"
        else:
            current_student_id = student_id
            similarity_list = create_similarity_list(student_id, reviews, 1)#hiperparametr tutaj
            grade = predict(movie_id, similarity_list)
            out_line = task_id + ";" + student_id + ";" + movie_id + ";" + grade + "\n"

    write(out_line, "output.csv")
    #exit(0)
        
        
    
        
        
        
    
