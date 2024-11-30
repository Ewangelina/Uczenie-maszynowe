import pandas as pd

def write(student_reviews, student_no):
    if not student_no == -1:
        o = open("student_reviews.csv", "a")
        line = str(student_no) + ";" + ";".join(student_reviews)
        line = line + "\n"
        o.write(line)
        o.close()

student_reviews = []
for i in range(200):
    student_reviews.append("X")

f = open(".\\train.csv")
last_student = -1

for data_line in f:
    garbage, student_no, movie_no, grade = data_line.split(";")
    movie_no = int(movie_no)
    grade = str(grade[:-1])
    student_no = str(student_no)

    if student_no == last_student:
        student_reviews[movie_no-1] = grade
    else:
        write(student_reviews, last_student)
        for i in range(200):
            student_reviews[i] = "X"
        last_student = student_no
        student_reviews[movie_no-1] = grade
        
    
f.close()



    
