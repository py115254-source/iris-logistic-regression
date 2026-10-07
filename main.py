from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# Encode target labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)


# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Iris Logistic Regression Classification")
print("---------------------------------------")
print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))