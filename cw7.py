import numpy as np
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split

def normalize(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

def euclidean(a, b):
    return np.sqrt(np.sum((a - b)**2, axis=1))

def manhattan(a, b):
    return np.sum(np.abs(a - b), axis=1)

def knn_predict(X_train, y_train, x, k, metric):
    if metric == "euclidean":
        d = euclidean(X_train, x)
    else:
        d = manhattan(X_train, x)
    idx = np.argsort(d)[:k]
    vals, counts = np.unique(y_train[idx], return_counts=True)
    return vals[np.argmax(counts)]

def evaluate(X_train, X_test, y_train, y_test, k, metric):
    preds = []
    for x in X_test:
        preds.append(knn_predict(X_train, y_train, x.reshape(1, -1), k, metric))
    return np.mean(np.array(preds) == y_test)

def run_experiment(dataset_loader):
    data = dataset_loader()
    X = normalize(data.data)
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, shuffle=True)
    ks = [1, 3, 5, 7]
    metrics = ["euclidean", "manhattan"]
    results = []
    for k in ks:
        for m in metrics:
            acc = evaluate(X_train, X_test, y_train, y_test, k, m)
            results.append((k, m, round(acc*100, 2)))
    return results

iris_results = run_experiment(load_iris)
wine_results = run_experiment(load_wine)

print("Iris results:")
for r in iris_results:
    print(r)

print("\nWine results:")
for r in wine_results:
    print(r)
