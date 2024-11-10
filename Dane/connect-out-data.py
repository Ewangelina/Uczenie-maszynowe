import requests
import json

def write(line, file):
    o = open(file, "a")
    o.write(line)
    o.close()


f = open(".\\task.csv")
last_student = -1
student_output = open(".\\movie.csv")

for task_line in f:
    task = task_line.split(";")
    student_no = task[1]
    if student_no == last_student:
        student_line = student_output.readline()
        task[3] = student_line.split(";")[1]
    else:
        student_output.close()
        filename = ".\\output\\" + str(student_no) + ".csv"
        student_output = open(filename)
        last_student = student_no
        student_line = student_output.readline()
        task[3] = student_line.split(";")[1]
        
    outline = ";".join(task)
    write(outline, "full_output.csv")
    
f.close()



    
