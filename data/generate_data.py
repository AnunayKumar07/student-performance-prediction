import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000
attendance = np.random.uniform(0, 100, n_samples)  # Attendance percentage
past_marks = np.random.uniform(0, 100, n_samples)  # Past marks
activities_score = np.random.uniform(0, 10, n_samples)  # Activities score (0-10)

# Risk label: At risk if attendance < 50 or past_marks < 40 or activities_score < 3
risk_label = ((attendance < 50) | (past_marks < 40) | (activities_score < 3)).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Attendance': attendance,
    'Past_Marks': past_marks,
    'Activities_Score': activities_score,
    'Risk_Label': risk_label
})

# Save to CSV
df.to_csv('data/student_data.csv', index=False)
print("Dataset generated and saved to data/student_data.csv")