import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("data/disease_dataset.csv")

# Features and labels
X = data[["Symptom1", "Symptom2", "Symptom3"]]
y = data["Disease"]

# Encode features and labels
feature_encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    feature_encoders[col] = le

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Prediction function
def predict_disease(symptoms):
    input_df = pd.DataFrame([symptoms])
    for col in input_df.columns:
        if col in feature_encoders:
            input_df[col] = feature_encoders[col].transform(input_df[col])
    prediction = model.predict(input_df)
    disease_name = label_encoder.inverse_transform(prediction)
    return disease_name[0]

# Example usage
if __name__ == "__main__":
    print("Enter your symptoms (all lowercase):")
    s1 = input("Symptom 1: ")
    s2 = input("Symptom 2: ")
    s3 = input("Symptom 3: ")
    symptoms = {"Symptom1": s1, "Symptom2": s2, "Symptom3": s3}
    result = predict_disease(symptoms)
    print(f"Predicted Disease: {result}")
