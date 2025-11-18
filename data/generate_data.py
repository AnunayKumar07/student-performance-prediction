import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 1000

# Features
total_marks = np.random.randint(0, 501, n_samples) # 0 to 500 inclusive
total_absences = np.random.randint(0, 31, n_samples) # 0 to 30 inclusive
supplementary_exams = np.random.randint(0, 6, n_samples) # 0 to 5 inclusive

# For categorical yes/no: encode 1 (Yes), 0 (No)
family_support = np.random.choice([0, 1], n_samples, p=[0.3,0.7])
extra_curricular = np.random.choice([0, 1], n_samples, p=[0.4,0.6])
wants_higher_education = np.random.choice([0, 1], n_samples, p=[0.5,0.5])
internet_access = np.random.choice([0, 1], n_samples, p=[0.2,0.8])

# Define Risk_Label using some logic
risk_label = (
    (total_marks < 250) |
    (total_absences > 20) |
    (supplementary_exams > 2) |
    (family_support == 0) |
    (internet_access == 0)
).astype(int)

df = pd.DataFrame({
    "total_marks": total_marks,
    "total_absences": total_absences,
    "supplementary_exams": supplementary_exams,
    "family_support": family_support,
    "extra_curricular": extra_curricular,
    "wants_higher_education": wants_higher_education,
    "internet_access": internet_access,
    "Risk_Label": risk_label
})

df.to_csv("data/student_data.csv", index=False)
print("Synthetic dataset generated to data/student_data.csv")