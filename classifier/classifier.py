import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from lazypredict.Supervised import LazyClassifier

data = pd.read_csv("diabetes.csv")
### Visualization raw data
profile = ProfileReport(data, title="Diabetes Report", explorative=True)
profile.to_file("report.html")

# Data split
target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Data processing
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# # Train model
model = RandomForestClassifier(n_estimators=200, criterion="gini", random_state=100)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print(classification_report(y_test, y_predict))

### Apply GridSearchcv for SVC
params = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": [0.001, 0.01, 0.1]
}

grid_search = GridSearchCV(estimator=SVC(random_state=100), param_grid=params, cv=4, scoring="recall", verbose=2)
grid_search.fit(x_train, y_train)
print(grid_search.best_estimator_)
print(grid_search.best_score_)
print(grid_search.best_params_)
y_predict = grid_search.predict(x_test)
print(classification_report(y_test, y_predict))

### Apply LazyCladdifier
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(x_train, x_test, y_train, y_test)
# Since `predictions` is a dataframe, extract the relevant column for the best model
print(predictions.head())




