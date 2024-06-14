# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Handle missing values using median imputation
imputer = SimpleImputer(strategy='median')
train_df['Arrival Delay in Minutes'] = imputer.fit_transform(train_df[['Arrival Delay in Minutes']])
test_df['Arrival Delay in Minutes'] = imputer.transform(test_df[['Arrival Delay in Minutes']])

# Encode categorical variables using LabelEncoder
label_encoders = {}
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for col in categorical_columns:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    if col != 'satisfaction':
        test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# Normalize/Scale numerical features using StandardScaler
scaler = StandardScaler()
numerical_columns = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
columns_to_remove = ['Unnamed: 0', 'id', 'satisfaction']
for col in columns_to_remove:
    if col in numerical_columns:
        numerical_columns.remove(col)

train_df[numerical_columns] = scaler.fit_transform(train_df[numerical_columns])
test_df[numerical_columns] = scaler.transform(test_df[numerical_columns])

# Plot distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='satisfaction', data=train_df)
plt.title('Distribution of Satisfaction')
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(16, 12))
correlation_matrix = train_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Split the training data into train and validation sets
X = train_df.drop(columns=['Unnamed: 0', 'id', 'satisfaction'])
y = train_df['satisfaction']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Random Forest classifier
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = rf_model.predict(X_val)

# Evaluate the model
conf_matrix = confusion_matrix(y_val, y_val_pred)
class_report = classification_report(y_val, y_val_pred)
accuracy = accuracy_score(y_val, y_val_pred)

# Print evaluation metrics
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
print("\nAccuracy Score:", accuracy)
