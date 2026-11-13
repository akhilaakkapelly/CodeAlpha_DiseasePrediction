# CodeAlpha_DiseasePrediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Sample health dataset
data = {
    "fever": [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
    "cough": [1, 1, 0, 1, 0, 0, 1, 1, 1, 0],
    "headache": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    "fatigue": [1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    "disease": [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop("disease", axis=1)
y = df["disease"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Disease Prediction Model")
print("Accuracy:", accuracy * 100, "%")

# New sample prediction
new_patient = [[1, 1, 0, 1]]
result = model.predict(new_patient)

print("Model Output:",
      "Positive class" if result[0] == 1 else "Negative class")
