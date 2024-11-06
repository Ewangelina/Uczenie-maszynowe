from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from kNN import kNN

def find_best_k(X, y, k_values=range(1, 21)):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    errors = []

    for k in k_values:
        print(f"Testing k={k}")
        knn = kNN(k=k)
        fold_errors = []

        for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
            print(f"  Fold {fold} for k={k}")
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]

            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)
            print(f"    MSE for fold {fold} and k={k}: {mse}")
            fold_errors.append(mse)

        avg_error = np.mean(fold_errors)
        print(f"Average error for k={k}: {avg_error}")
        errors.append((k, avg_error))

    best_k = min(errors, key=lambda x: x[1])[0]
    print(f"Best k found: {best_k}")
    return best_k

