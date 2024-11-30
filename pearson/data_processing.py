

def load_reviews(file_path=".\\..\\Dane\\student_reviews.csv"):
    f = open(file_path)
    result = []
    idiots = []
    for line in f:
        split_line = line.split(";")
        if is_user_stupid(split_line):
            idiots.append(split_line)
        else:
            result.append(split_line)
            
    f.close()
    return result, idiots

def is_user_stupid(row):
    usual_grade = row[1]
    for i in range(2, len(row)):
        if row[i] == usual_grade:
            return False

    return True
        
