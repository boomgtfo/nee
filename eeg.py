import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace

# Read TSV file
df = pd.read_csv('./sub-s1_task-PerceiveImagine_events.tsv', delimiter='\t')

# Display the first few rows of data
print(df.head())

# Extract relevant columns
data = df[['onset', 'duration', 'sample']]

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Dynamically find the label column name
column_names = df.columns.tolist()

label_column_name = 'value'

# Raise an error if the label column name is not found
if label_column_name not in df.columns:
    raise ValueError("Label column not found")

# Get labels
labels = df[label_column_name]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2, random_state=42)

# Use Gaussian Naive Bayes model
gnb = GaussianNB()

# Use StratifiedKFold for 5-fold cross-validation
strat_kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
scores = cross_val_score(gnb, X_train, y_train, cv=strat_kfold)  # 5-fold cross-validation
mean_accuracy = np.mean(scores)

print(f"Model accuracy (average of 5-fold cross-validation): {mean_accuracy:.2f}")

# Laplace smoothing: Apply Laplace smoothing to the model
gnb.set_params(var_smoothing=1e-9)  # Set Laplace smoothing parameter to avoid zero frequencies

# Train the model
gnb.fit(X_train, y_train)

# Predict
y_pred = gnb.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

# Display classification report
class_report = classification_report(y_test, y_pred)
print(f"Classification Report:\n{class_report}")

# Fit a Gaussian distribution
mu, std = np.mean(data_scaled[:, 0]), np.std(data_scaled[:, 0])

# Generate some Gaussian-distributed data points
x = np.linspace(mu - 3*std, mu + 3*std, 100)
p = norm.pdf(x, mu, std)

# Plot the Gaussian distribution
plt.figure()
plt.plot(x, p, 'k-', linewidth=2)
plt.title('Fitted Gaussian Distribution')
plt.show()

# Fit a Laplace distribution
loc, scale = laplace.fit(data_scaled[:, 0])

# Generate some Laplace-distributed data points
x_laplace = np.linspace(loc - 3*scale, loc + 3*scale, 100)
p_laplace = laplace.pdf(x_laplace, loc, scale)

# Plot the Laplace distribution
plt.figure()
plt.plot(x_laplace, p_laplace, 'k-', linewidth=2)
plt.title('Fitted Laplace Distribution')
plt.show()
