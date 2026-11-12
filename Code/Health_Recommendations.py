import joblib
import pandas as pd
from pathlib import Path
project_folder = Path(__file__).resolve().parent.parent
model_path = project_folder / "Results" / "final_diabetes_model.pkl"
model_artifact = joblib.load(model_path)
model = model_artifact["model"]
threshold = model_artifact["threshold"]
features = model_artifact["features"]
input_guides = {
    "HighBP": "0 = No, 1 = Yes",
    "HighChol": "0 = No, 1 = Yes",
    "CholCheck": "0 = No, 1 = Yes",
    "Smoker": "0 = No, 1 = Yes",
    "Stroke": "0 = No, 1 = Yes",
    "HeartDiseaseorAttack": "0 = No, 1 = Yes",
    "PhysActivity": "0 = No, 1 = Yes",
    "Fruits": "0 = No, 1 = Yes",
    "Veggies": "0 = No, 1 = Yes",
    "HvyAlcoholConsump": "0 = No, 1 = Yes",
    "AnyHealthcare": "0 = No, 1 = Yes",
    "NoDocbcCost": "0 = No, 1 = Yes",
    "DiffWalk": "0 = No, 1 = Yes",
    "Sex": "0 = Female, 1 = Male",
    "GenHlth": "1 = Excellent, 2 = Very Good, 3 = Good, 4 = Fair, 5 = Poor",
    "Age": "1 = 18-24, 2 = 25-29, 3 = 30-34, 4 = 35-39, 5 = 40-44, 6 = 45-49, 7 = 50-54, 8 = 55-59, 9 = 60-64, 10 = 65-69, 11 = 70-74, 12 = 75-79, 13 = 80+",
    "Education": "1 = Never attended/Kindergarten, 2 = Grades 1-8, 3 = Grades 9-11, 4 = Grade 12/GED, 5 = College 1-3 years, 6 = College 4+ years",
    "Income": "1 = Below $10,000, 2 = $10,000-$15,000, 3 = $15,000-$20,000, 4 = $20,000-$25,000, 5 = $25,000-$35,000, 6 = $35,000-$50,000, 7 = $50,000-$75,000, 8 = $75,000+",
    "BMI": "Enter numerical BMI value",
    "MentHlth": "Number of days of poor mental health in the last 30 days (0-30)",
    "PhysHlth": "Number of days of poor physical health in the last 30 days (0-30)"
}
patient_data = {}
print("\nDIABETES RISK ASSESSMENT")
print("=" * 40)
print("Enter the patient's health information:\n")
for feature in features:
    while True:
        print(f"\n{feature}")
        print(input_guides.get(feature, "Enter numerical value"))
        try:
            value = float(input("Your input: "))
            patient_data[feature] = value
            break
        except ValueError:
            print("Please enter a valid number.")
patient_df = pd.DataFrame([patient_data], columns=features)
probability = model.predict_proba(patient_df)[0][1]
prediction = int(probability >= threshold)
print("\nASSESSMENT RESULT")
print("=" * 40)
print(f"Diabetes risk score: {probability * 100:.2f}%")
if prediction == 1:
    print("\nModel indicates a potential diabetes risk.")
    print("\nGeneral Health Precautions:")
    print("1. Maintain a balanced and healthy diet.")
    print("2. Engage in regular physical activity.")
    print("3. Maintain a healthy body weight.")
    print("4. Avoid smoking.")
    print("5. Monitor blood glucose and consult a doctor.")
else:
    print("\nModel did not indicate diabetes risk.")
    print("Continue maintaining a healthy lifestyle.")
print("\nDisclaimer: This prediction is not a medical diagnosis.")
print("Consult a healthcare professional for proper testing.")