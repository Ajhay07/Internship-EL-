import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:\Users\Ajhay\internship\task5\heart.csv')  

print(df.head())
print(df.isnull().sum())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True)
plt.title("Decision Tree")
plt.show()

train_acc = dt.score(X_train, y_train)
test_acc = dt.score(X_test, y_test)
print(f"Decision Tree - Train Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"Random Forest - Test Accuracy: {rf_acc:.2f}")

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', title="Feature Importances")
plt.show()

cv_scores = cross_val_score(rf, X, y, cv=5)
print(f"Cross-validated accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std():.2f})")
