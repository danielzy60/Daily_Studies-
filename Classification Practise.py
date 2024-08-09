#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,mean_squared_error
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df_s = pd.read_csv("pc2_small_train_data_v1.csv")
df_l = pd.read_csv("pc2_large_train_data_v1.csv")
df_test = pd.read_csv("pc2_test_without_response_variable_v1.csv")


# In[3]:


df_s['P_Under_19500'] = np.where(df_s['price'] < 19500, 1, 0)
df_l['P_Under_19500'] = np.where(df_l['price'] < 19500, 1, 0)
df_s['nyears'] = 2024 - df_s['year']
df_l['nyears'] = 2024 - df_l['year']
df_test['nyears'] = 2024 - df_test['year']


# In[4]:


df_c = pd.concat([df_s, df_l])

# Remove duplicate rows
df_c = df_c.drop_duplicates()


# ### data cleaning 

# In[5]:


# Check for missing values
print(df_c.isnull().sum())
print(df_test.isnull().sum())

# Fill missing numerical values with the mean
numerical_cols_c = df_c.select_dtypes(include=['float64', 'int64']).columns
numerical_cols_test = df_test.select_dtypes(include=['float64', 'int64']).columns

df_c[numerical_cols_c] = df_c[numerical_cols_c].fillna(df_c[numerical_cols_c].mean())
df_test[numerical_cols_test] = df_test[numerical_cols_test].fillna(df_test[numerical_cols_test].mean())



# ### split major_options into binary, since there are different elements in each cell of major_options

# In[6]:


import ast

# Function to safely evaluate strings to lists
def safe_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
    elif isinstance(val, list):
        return val
    return []


# Apply the function to safely evaluate strings to lists
df_c['major_options'] = df_c['major_options'].apply(safe_eval)
df_test['major_options'] = df_test['major_options'].apply(safe_eval)

# Create the option_count feature for both dataframes
df_c['option_count'] = df_c['major_options'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df_test['option_count'] = df_test['major_options'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Create binary columns for each unique option in df_c
options = df_c['major_options'].explode().dropna().unique()
for option in options:
    df_c[option] = df_c['major_options'].apply(lambda x: 1 if isinstance(x, list) and option in x else 0)
    df_test[option] = df_test['major_options'].apply(lambda x: 1 if isinstance(x, list) and option in x else 0)

# Drop the original major_options column from both dataframes
df_c = df_c.drop(columns=['major_options'])
df_test = df_test.drop(columns=['major_options'])

# Calculate the correlation of each option with the price in df_c
correlation = df_c.corr()['price'].sort_values(ascending=False)

# Filter out only the correlations of the major options
correlation_major_options = correlation[options]

# Select options with high positive correlation (e.g., correlation > 0.2)
high_positive_corr_options = correlation_major_options[correlation_major_options > 0.2].index

# Create new features that count the number of high correlation options present in both dataframes
df_c['high_price_related'] = df_c[high_positive_corr_options].sum(axis=1)
df_test['high_price_related'] = df_test[high_positive_corr_options].sum(axis=1)





# ### dealing with missing values for categorical

# In[7]:


# Fill missing categorical values with the mode
categorical_cols_c = df_c.select_dtypes(include=['object']).columns
categorical_cols_test = df_test.select_dtypes(include=['object']).columns

df_c[categorical_cols_c] = df_c[categorical_cols_c].fillna(df_c[categorical_cols_c].mode().iloc[0])
df_test[categorical_cols_test] = df_test[categorical_cols_test].fillna(df_test[categorical_cols_test].mode().iloc[0])


# ### remove unit for numerical value

# In[8]:


def remove_units(value):
    try:
        return float(value.replace(' in', '').replace(' cm', '').replace(' ft', '').replace(' kg', ''))
    except:
        return np.nan

# List of columns with units
columns_with_units = ['back_legroom', 'front_legroom', 'height', 'length', 'wheelbase', 'width']

# Apply the function to the relevant columns for df_c
for col in columns_with_units:
    df_c[col] = df_c[col].apply(remove_units)

# Apply the function to the relevant columns for df_test
for col in columns_with_units:
    df_test[col] = df_test[col].apply(remove_units)


# ### dealing with missing value for numerical data (replace with mode)

# In[9]:


# Replace missing values in all numerical columns with the mode
numerical_cols = df_c.select_dtypes(include=[np.number]).columns
numerical_cols_test = df_test.select_dtypes(include=[np.number]).columns

for col in numerical_cols:
    if df_c[col].isnull().any():
        mode_value = df_c[col].mode()
        if not mode_value.empty:
            mode_value = mode_value[0]
            df_c[col].fillna(mode_value, inplace=True)
        else:
            print(f"No mode found for column: {col}")



# Check for any remaining missing values
missing_values = df_c[numerical_cols].isnull().sum()
print("Missing values in each column after imputation:")
print(missing_values[missing_values > 0])




# In[10]:


categorical_cols = df_c.select_dtypes(include=['object']).columns
# Replace missing values in all categorical columns with the mode
for col in categorical_cols:
    mode_value = df_c[col].mode()[0]
    df_c[col].fillna(mode_value, inplace=True)


# In[12]:


X_s = df_c.drop(['price', 'year', 'P_Under_19500'], axis=1)
Y_s = df_c['P_Under_19500']


# In[13]:


X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_s, Y_s, test_size=0.2, random_state=42)



# In[14]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Identify numerical and categorical columns
numerical_cols = X_train_s.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train_s.select_dtypes(include=['object', 'category']).columns

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Define the pipeline with the best parameters found
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Fit the model
pipeline.fit(X_train_s, y_train_s)

# Predict on the test set
y_pred_s = pipeline.predict(X_test_s)

# Evaluate the model
accuracy = accuracy_score(y_test_s, y_pred_s)
print(f'Accuracy: {accuracy}')

print('Classification Report:')
print(classification_report(y_test_s, y_pred_s))



# In[15]:


# Compute ROC AUC score
y_pred_proba_s = pipeline.predict_proba(X_test_s)[:, 1]
roc_auc = roc_auc_score(y_test_s, y_pred_proba_s)
print(f'ROC AUC Score: {roc_auc}')

# Display feature importances
importances = pipeline.named_steps['classifier'].feature_importances_

# Get the feature names from the OneHotEncoder
onehot_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
onehot_feature_names = onehot_encoder.get_feature_names_out(categorical_cols)
features = numerical_cols.tolist() + onehot_feature_names.tolist()

feature_importances = pd.Series(importances, index=features).sort_values(ascending=False)
print('Top 20 Feature Importances:')
print(feature_importances.head(20))

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances.head(20), y=feature_importances.head(20).index)
plt.title('Top 20 Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


# ### some feature eng based on the above result 

# In[11]:


# Ensure 'wheel_system_display' and 'trim_name' are treated as categorical for interaction term creation
df_c['wheel_system_display'] = df_c['wheel_system_display'].astype('category')
df_c['trim_name'] = df_c['trim_name'].astype('category')
df_test['wheel_system_display'] = df_test['wheel_system_display'].astype('category')
df_test['trim_name'] = df_test['trim_name'].astype('category')


# In[12]:


# Create interaction features for df_c
df_c['horsepower_nyears'] = df_c['horsepower'] * df_c['nyears']
df_c['mileage_nyears'] = df_c['mileage'] / df_c['nyears']
df_c['wheel_trim'] = df_c['wheel_system_display'].cat.codes * df_c['trim_name'].cat.codes


# In[13]:


# Create interaction features for df_test
df_test['horsepower_nyears'] = df_test['horsepower'] * df_test['nyears']
df_test['mileage_nyears'] = df_test['mileage'] / df_test['nyears']
df_test['wheel_trim'] = df_test['wheel_system_display'].cat.codes * df_test['trim_name'].cat.codes


# In[14]:


# Scatter plot of price vs horsepower with polynomial regression line
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x=df_c['horsepower'], y=df_c['price'], alpha=0.3)
plt.title('Price vs Horsepower')
plt.xlabel('Horsepower')
plt.ylabel('Price')

# Scatter plot of price vs nyears with polynomial regression line
plt.subplot(1, 2, 2)
sns.scatterplot(x=df_c['nyears'], y=df_c['price'], alpha=0.3)
plt.title('Price vs Nyears')
plt.xlabel('Nyears')
plt.ylabel('Price')

plt.tight_layout()
plt.show()


# In[15]:


### polynomial to capture non linearity 


# In[16]:


# Create polynomial features for df_c
df_c['horsepower^2'] = df_c['horsepower'] ** 2
df_c['horsepower^3'] = df_c['horsepower'] ** 3
df_c['nyears^2'] = df_c['nyears'] ** 2
df_c['nyears^3'] = df_c['nyears'] ** 3

# Create polynomial features for df_test
df_test['horsepower^2'] = df_test['horsepower'] ** 2
df_test['horsepower^3'] = df_test['horsepower'] ** 3
df_test['nyears^2'] = df_test['nyears'] ** 2
df_test['nyears^3'] = df_test['nyears'] ** 3

# Print the first few rows of the updated df_c and df_test to verify
print("df_c with polynomial features:")
print(df_c[['horsepower', 'horsepower^2', 'horsepower^3', 'nyears', 'nyears^2', 'nyears^3']].head())

print("\ndf_test with polynomial features:")
print(df_test[['horsepower', 'horsepower^2', 'horsepower^3', 'nyears', 'nyears^2', 'nyears^3']].head())


# In[17]:


# Define age categories
def categorize_age(nyears):
    if nyears <= 5:
        return 'new'
    elif nyears <= 10:
        return 'mid_age'
    else:
        return 'old'

df_c['age_category'] = df_c['nyears'].apply(categorize_age)
df_test['age_category'] = df_test['nyears'].apply(categorize_age)


# In[18]:


# Define brand origins
europe_brands = ['Audi', 'BMW', 'Mercedes-Benz', 'Volkswagen', 'Porsche', 'Volvo', 'Jaguar', 'Mini', 'Land Rover', 'Ferrari', 'Maserati', 'Bentley', 'Rolls-Royce']
asia_brands = ['Toyota', 'Honda', 'Nissan', 'Hyundai', 'Kia', 'Mazda', 'Subaru', 'Mitsubishi', 'Lexus', 'Infiniti', 'Acura', 'Suzuki', 'Genesis', 'Isuzu']
north_america_brands = ['Ford', 'Chevrolet', 'GMC', 'Cadillac', 'Jeep', 'Dodge', 'Chrysler', 'Buick', 'Ram', 'Lincoln', 'Tesla', 'Hummer', 'Pontiac', 'Saturn']

# Function to determine the origin based on the brand
def get_origin(make_name):
    if make_name in europe_brands:
        return 'Europe'
    elif make_name in asia_brands:
        return 'Asia'
    elif make_name in north_america_brands:
        return 'North America'
    else:
        return 'Others'

# Assuming df_c is the DataFrame you're working with
# Map make_name to origin
df_c['origin'] = df_c['make_name'].apply(get_origin)
df_test['origin'] = df_test['make_name'].apply(get_origin)

# Check for any missing values in the new column
print(df_c['origin'].isnull().sum())
print(df_test['origin'].isnull().sum())

# Fill missing values with 'Unknown'
df_c['origin'].fillna('Unknown', inplace=True)
df_test['origin'].fillna('Unknown', inplace=True)


# ### rerun after featuring

# In[19]:


# Identify categorical columns
categorical_cols = df_c.select_dtypes(include=['object']).columns

# Replace missing values in all categorical columns with the mode
for col in categorical_cols:
    mode_value = df_c[col].mode()[0]
    df_c[col].fillna(mode_value, inplace=True)


# In[25]:


X_s = df_c.drop(['price', 'P_Under_19500'], axis=1)
Y_s = df_c['P_Under_19500']


# In[26]:


X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_s, Y_s, test_size=0.2, random_state=42)


# In[27]:


# Identify numerical and categorical columns
numerical_cols = X_train_s.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train_s.select_dtypes(include=['object', 'category']).columns

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Define the pipeline with the best parameters found
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Fit the model
pipeline.fit(X_train_s, y_train_s)

# Predict on the test set
y_pred_s = pipeline.predict(X_test_s)

# Evaluate the model
accuracy = accuracy_score(y_test_s, y_pred_s)
print(f'Accuracy: {accuracy}')

print('Classification Report:')
print(classification_report(y_test_s, y_pred_s))

# Compute ROC AUC score
y_pred_proba_s = pipeline.predict_proba(X_test_s)[:, 1]
roc_auc = roc_auc_score(y_test_s, y_pred_proba_s)
print(f'ROC AUC Score: {roc_auc}')



# In[28]:


### try KNN


# In[29]:


# Prepare the feature matrix and target vector
X = df_c.drop(columns=['price', 'P_Under_19500'])
y = df_c['P_Under_19500']

# Split the data into training and testing sets
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y, test_size=0.2, random_state=42)


# In[30]:


categorical_features = df_c.select_dtypes(include=['object']).columns

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Identify numerical and categorical columns
numerical_cols = X_train_s.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train_s.select_dtypes(include=['object', 'category']).columns

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)


# Define the model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', KNeighborsClassifier())
])

# Train the model
model_pipeline.fit(X_train_s, y_train_s)

# Predict on the test set
y_pred_s = model_pipeline.predict(X_test_s)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test_s, y_pred_s))
print("Confusion Matrix:\n", confusion_matrix(y_test_s, y_pred_s))
print("Classification Report:\n", classification_report(y_test_s, y_pred_s))
# In[ ]:


### try Naives Bayes


# In[ ]:


# Initialize the Naive Bayes model
nb_model = GaussianNB()

# Train the model
nb_model.fit(X_train_s, y_train_s)

# Predict on the test set
y_pred_s = nb_model.predict(X_test_s)
y_pred_prob_s = nb_model.predict_proba(X_test_s)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test_s, y_pred_s)
classification_rep = classification_report(y_test_s, y_pred_s)
roc_auc = roc_auc_score(y_test_s, y_pred_prob_s)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_rep)
print(f"ROC AUC Score: {roc_auc}")


# ### we try all three base model, and find that tree has the best accuracy, now try to tune the tree and use one hot encoding
# Define the parameter grid for GridSearchCV
param_grid = {
    'classifier__criterion': ['entropy'],
    'classifier__splitter': ['random'],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [5, 10],
    'classifier__min_samples_leaf': [2, 4]
}

# Set up GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=10, n_jobs=-1, verbose=1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print(f'Best parameters found: {grid_search.best_params_}')
print(f'Best cross-validation score: {grid_search.best_score_}')

# Predict on the test set with the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Print the test accuracy
accuracy = best_model.score(X_test, y_test)
print(f'Test accuracy: {accuracy}')

# Display feature importances of the best model
importances = best_model.named_steps['classifier'].feature_importances_

# Get the feature names after one-hot encoding
onehot = best_model.named_steps['preprocessor'].transformers_[1][1]
onehot.fit(X_train[categorical_cols])
onehot_feature_names = onehot.get_feature_names_out(categorical_cols)
features = numerical_cols.tolist() + onehot_feature_names.tolist()

feature_importances = pd.Series(importances, index=features).sort_values(ascending=False)
print(feature_importances.head(20))
# In[ ]:





# 
# ### Best parameters found: {'classifier__criterion': 'entropy', 'classifier__max_depth': 20, 'classifier__min_samples_leaf': 4, 'classifier__min_samples_split': 10, 'classifier__splitter': 'random'}### after grid search: 

# In[20]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Prepare the feature matrix and target vector
X = df_c.drop(columns=['price', 'P_Under_19500'])
y = df_c['P_Under_19500']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

# Define preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# Define the pipeline with the best parameters found
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        criterion='entropy',
        max_depth=20,
        min_samples_leaf=4,
        min_samples_split=10,
        splitter='random',
        random_state=42
    ))
])

# Fit the model
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

print('Classification Report:')
print(classification_report(y_test, y_pred))

# Compute ROC AUC score
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f'ROC AUC Score: {roc_auc}')

# Display feature importances
importances = pipeline.named_steps['classifier'].feature_importances_

# Get the feature names from the OneHotEncoder
onehot_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
onehot_feature_names = onehot_encoder.get_feature_names_out(categorical_cols)
features = numerical_cols.tolist() + onehot_feature_names.tolist()

feature_importances = pd.Series(importances, index=features).sort_values(ascending=False)
print('Top 20 Feature Importances:')
print(feature_importances.head(20))

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances.head(20), y=feature_importances.head(20).index)
plt.title('Top 20 Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()


# In[21]:


X_test_final = df_test[X.columns].copy()
X_test_final.head()


# In[22]:


X_train.head()


# In[26]:


# Handle missing values in df_test
for col in df_test.columns:
    if df_test[col].isnull().sum() > 0:
        if df_test[col].dtype in ['int64', 'float64']:
            df_test[col].fillna(df_test[col].median(), inplace=True)
        else:
            df_test[col].fillna(df_test[col].mode()[0], inplace=True)

# Ensure the columns in df_test match those in X
X_test_final = df_test[X.columns].copy()

# Predict on df_test
y_test_pred = pipeline.predict(X_test_final)
y_test_pred_proba = pipeline.predict_proba(X_test_final)[:, 1]

# Output the predictions
df_test['P_Under_19500_Pred'] = y_test_pred
df_test['P_Under_19500_Proba'] = y_test_pred_proba

print(df_test[['model_name', 'P_Under_19500_Pred', 'P_Under_19500_Proba']].head())



# In[27]:


# Save the predictions to a CSV file
df_test['P_Under_19500_Pred'].to_csv('df_test_predictions.csv', index=False)


