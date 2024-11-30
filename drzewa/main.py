from data_processing import load_and_process_movies, load_train_data, load_task_data
from decisionTree import decisionTree
from randomForest import randomForest
from os import listdir
from os.path import join

def split_data(X, y, validation_ratio=0.2):
    """
    Split the data into training and validation sets.
    """
    split_index = int(len(X) * (1 - validation_ratio))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def evaluate_model(model, X_val, y_val):
    """
    Evaluate the model on validation data and return the mean squared error.
    """
    predictions = [int(model.useTree(x)) if hasattr(model, 'useTree') else int(model.predict(x)) for x in X_val]
    mse = sum((p - y) ** 2 for p, y in zip(predictions, y_val)) / len(y_val)
    return mse

def grid_search_decision_tree(X_train, y_train, X_val, y_val, output_path):
    """
    Perform grid search for the decision tree.
    """
    best_params = None
    best_mse = float('inf')

    # Define hyperparameter grid
    split_ratios = [0.6, 0.7, 0.8]
    min_splits = [2, 5, 10]
    
    for split_ratio in split_ratios:
        for min_split in min_splits:
            dt = decisionTree(output_path)
            dt.createTree(X_train, y_train, split_ratio, min_split)
            mse = evaluate_model(dt, X_val, y_val)
            if mse < best_mse:
                best_mse = mse
                best_params = (split_ratio, min_split)

    print(f"Best Decision Tree Params: {best_params}, MSE: {best_mse}")
    return best_params

def grid_search_random_forest(X_train, y_train, X_val, y_val, output_path):
    """
    Perform grid search for the random forest.
    """
    best_params = None
    best_mse = float('inf')

    n_estimators = [10, 13, 20]
    split_ratios = [0.6, 0.7, 0.8]
    min_splits = [2, 5, 10]
    
    for n_tree in n_estimators:
        for split_ratio in split_ratios:
            for min_split in min_splits:
                rf = randomForest(X_train, y_train, output_path, n_tree, split_ratio, min_split)
                mse = evaluate_model(rf, X_val, y_val)
                if mse < best_mse:
                    best_mse = mse
                    best_params = (n_tree, split_ratio, min_split)

    print(f"Best Random Forest Params: {best_params}, MSE: {best_mse}")
    return best_params

# Directory paths
train_dir = "Dane\\train\\"
task_dir = "Dane\\task\\"
tree_output_dir = "drzewa\\output\\tree"
forest_output_dir = "drzewa\\output\\forest\\"
tree_dir = "drzewa\\trees\\"

print("Preprocess: Loading and processing movie data...")
movie_df = load_and_process_movies()

# Iterate through all user files
all_train_files = [f for f in listdir(train_dir) if f.endswith('.csv')]

for file in all_train_files:
    print(f"Processing user file: {file}")

    # Paths for current user
    trainFilePath = join(train_dir, file)
    treeDestination = join(tree_dir, file)
    taskFilePath = join(task_dir, file)
    treeOutputFilePath = join(tree_output_dir, file)
    forestOutputFilePath = join(forest_output_dir, file)

    # Load training data for user
    X, y = load_train_data(movie_df, trainFilePath)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = split_data(X, y)

    # Grid search for Decision Tree
    best_dt_params = grid_search_decision_tree(X_train, y_train, X_val, y_val, treeDestination)

    # Train final Decision Tree
    dt = decisionTree(treeDestination)
    dt.createTree(X_train, y_train, *best_dt_params)

    # Load task data and make predictions
    task_df, X_task = load_task_data(movie_df, taskFilePath)
    y_task = [int(dt.useTree(row)) for row in X_task]

    # Save Decision Tree predictions
    task_df["Predicted Rating"] = y_task
    task_df[["User ID", "Predicted Rating"]].to_csv(treeOutputFilePath, index=False, sep=";", header=False)

    # Grid search for Random Forest
    best_rf_params = grid_search_random_forest(X_train, y_train, X_val, y_val, treeDestination)

    # Train final Random Forest
    n_tree, split_ratio, min_split = best_rf_params
    rf = randomForest(X_train, y_train, treeDestination, n_tree, split_ratio, min_split)
    y_task = [int(rf.predict(row)) for row in X_task]

    # Save Random Forest predictions
    task_df["Predicted Rating"] = y_task
    task_df[["User ID", "Predicted Rating"]].to_csv(forestOutputFilePath, index=False, sep=";", header=False)

print("All user files processed. TASK DONE.")
