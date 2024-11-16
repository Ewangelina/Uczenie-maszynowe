from data_processing import load_and_process_movies, load_train_data, load_task_data
from decisionTree import decisionTree
from os import listdir
from os.path import isfile, join

def addNode(string, file):
    o = open(file, "a")
    string = string + "\n"
    o.write(string)
    o.close()

#all_train_files = listdir(".\\..\\Dane\\train")
print("Preprocess: Loading and processing movie data...")
movie_df = load_and_process_movies()

#for file in all_train_files:
if True:
    file = "307.csv"
    trainFilePath = ".\\..\\Dane\\train\\" + file
    treeDestination = ".\\trees\\" + file
    print(f"Loading training data for student {file}")
    X_train, y_train = load_train_data(movie_df, trainFilePath)

    dt = decisionTree(treeDestination)
    dt.createTree(X_train, y_train, 0.9, 1)

    taskFilePath = ".\\..\\Dane\\task\\" + file
    outputFilePath = ".\\output\\" + file
    task_df, X_task = load_task_data(movie_df, taskFilePath)
    y_task = []
    for row in X_task:
        y_task.append(int(dt.useTree(row)))
     
    task_df["Predicted Rating"] = y_task
    task_df[["User ID", "Predicted Rating"]].to_csv(outputFilePath, index=False, sep=";", header=False)


print("TASK DONE")
