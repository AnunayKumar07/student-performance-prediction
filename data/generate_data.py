import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

# Existing features
attendance = np.random.uniform(0, 100, n_samples)
past_marks = np.random.uniform(0, 100, n_samples)
activities_score = np.random.uniform(0, 10, n_samples)

# New features
total_marks = np.random.uniform(0, 1000, n_samples)  # Total marks out of 1000
total_absences = np.random.randint(0, 50, n_samples)  # Total absences
supplementary_exams = np.random.randint(0, 5, n_samples)  # Number of fail subjects
internet_access = np.random.randint(1, 6, n_samples)  # 1-5 scale
family_support = np.random.randint(1, 6, n_samples)  # 1-5 scale

# Risk label: Enhanced logic
risk_label = (
    (attendance < 50) | (past_marks < 40) | (activities_score < 3) |
    (total_absences > 20) | (supplementary_exams > 2) |
    (internet_access < 3) | (family_support < 3)
).astype(int)

df = pd.DataFrame({
    'Attendance': attendance,
    'Past_Marks': past_marks,
    'Activities_Score': activities_score,
    'Total_Marks': total_marks,
    'Total_Absences': total_absences,
    'Supplementary_Exams': supplementary_exams,
    'Internet_Access': internet_access,
    'Family_Support': family_support,
    'Risk_Label': risk_label
})

df.to_csv('data/student_data.csv', index=False)
print("Enhanced dataset generated and saved.")