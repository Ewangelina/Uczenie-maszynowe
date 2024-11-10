import requests
import json

def write(line, file):
    o = open(file, "a")
    o.write(line)
    o.close()

#f = open(".\\train.csv")
f = open(".\\task.csv")

for line in f:
    student_no = line.split(";")[1]
#    file = ".\\train\\" + student_no + ".csv"
    file = ".\\task\\" + student_no + ".csv"
    write(line, file)
    
f.close()



    
