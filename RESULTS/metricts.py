import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your file
df = pd.read_csv('predictions.csv')

# Filter for the 'test' split only
test_df = df[df['split'] == 'test']

# Define true and predicted values
y_true = test_df['cbm_true']
y_pred = test_df['cbm_pred']

# Calculate metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"Metrics for Test Split:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")