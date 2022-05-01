from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from joblib import dump

def save_model():
    iris = load_iris()
    X = iris.data
    y = iris.target
    model = LogisticRegression(max_iter = 1e3).fit(X, y)
    dump(model, 'model.joblib')

if __name__ == '__main__':
    save_model()
# print(iris.target_names)
# ['setosa' 'versicolor' 'virginica']