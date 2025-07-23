import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# 1. Load data
df = pd.read_csv('insurance_extended_40k.csv')

# 2. Feature engineering
df['bmi_smoker']      = df['bmi'] * df['smoker'].map({'yes':1,'no':0})
df['age_chronic']     = df['age'] * df['chronic_disease']
df['bmi_age']         = df['bmi'] * df['age']
df['children_smoker'] = df['children'] * df['smoker'].map({'yes':1,'no':0})

X = df.drop('charges', axis=1)
y = df['charges']

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocessor + model pipeline
categorical = ['sex','smoker','region','exercise_level','alcohol_consumption',
               'chronic_disease','family_history','married','occupation_type']
numerical   = ['age','bmi','children','bmi_smoker','age_chronic','bmi_age','children_smoker']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])
pipeline = Pipeline([
    ('preproc', preprocessor),
    ('model', GradientBoostingRegressor(n_estimators=200, random_state=42))
])

# 5. Train
pipeline.fit(X_train, y_train)

# 6. Save
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("✅ model.pkl saved")
