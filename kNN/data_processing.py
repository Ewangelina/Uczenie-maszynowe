import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

def load_and_process_movies(file_path="Dane\\moviedeets.csv"):
    # Define column names based on data structure
    movie_columns = ["ID", "Budget", "Revenue", "Vote Count", "Vote Average", "Genres", "Production Countries", "Original Language"]

    # Load movie data
    df = pd.read_csv(file_path, delimiter=';', header=None, names=movie_columns)

    # Numeric conversions
    df["Budget"] = pd.to_numeric(df["Budget"], errors='coerce').fillna(0)
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors='coerce').fillna(0)
    df["Vote Count"] = pd.to_numeric(df["Vote Count"], errors='coerce').fillna(0)
    df["Vote Average"] = pd.to_numeric(df["Vote Average"], errors='coerce').fillna(0)

    # Split and encode categorical fields
    df["Genres"] = df["Genres"].str.split("#")
    df["Production Countries"] = df["Production Countries"].str.split("#")
    df["Original Language"] = df["Original Language"].str.split("#")

    mlb_genres = MultiLabelBinarizer()
    mlb_countries = MultiLabelBinarizer()
    mlb_language = MultiLabelBinarizer()

    genres_encoded = pd.DataFrame(mlb_genres.fit_transform(df["Genres"]), columns=mlb_genres.classes_, index=df.index)
    countries_encoded = pd.DataFrame(mlb_countries.fit_transform(df["Production Countries"]), columns=mlb_countries.classes_, index=df.index)
    language_encoded = pd.DataFrame(mlb_language.fit_transform(df["Original Language"]), columns=mlb_language.classes_, index=df.index)

    df = pd.concat([df, genres_encoded, countries_encoded, language_encoded], axis=1)
    df = df.drop(columns=["Genres", "Production Countries", "Original Language"])

    # Standardize numerical fields
    scaler = StandardScaler()
    df[["Budget", "Revenue", "Vote Count", "Vote Average"]] = scaler.fit_transform(df[["Budget", "Revenue", "Vote Count", "Vote Average"]])

    print("Preprocessing finished")
    return df

def load_train_data(movie_df, file_path="Dane\\train.csv"):
    ratings_columns = ["Row ID", "User ID", "Movie ID", "Rating"]
    ratings_df = pd.read_csv(file_path, delimiter=';', header=None, names=ratings_columns)

    # Merge with movie features based on Movie ID
    ratings_df = ratings_df.merge(movie_df, left_on="Movie ID", right_on="ID", how="left")
    ratings_df = ratings_df.drop(columns=["Row ID", "Movie ID", "ID"])

    X_train = ratings_df.drop(columns=["User ID", "Rating"]).values
    y_train = ratings_df["Rating"].values

    print("Train data loaded")
    return X_train, y_train

def load_task_data(movie_df, file_path="Dane\\task.csv"):
    task_columns = ["Row ID", "User ID", "Movie ID", "Rating"]
    task_df = pd.read_csv(file_path, delimiter=';', header=None, names=task_columns)

    task_df = task_df.merge(movie_df, left_on="Movie ID", right_on="ID", how="left")
    task_df = task_df.drop(columns=["Row ID", "Movie ID", "ID", "Rating"])

    X_task = task_df.drop(columns=["User ID"]).values

    print("Task data loaded")
    return task_df, X_task
