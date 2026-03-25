import pandas as pd
import numpy as np
# -------------------------------

# 1. DATASET

# -------------------------------
data = {
    "Hours":[1,2,3,4,5,6,7,8,9,10],
    "Sleep":[8,7,6,6,5,5,4,4,3,3],
    "Stress":[3,4,5,6,7,8,9,8,7,6],
    "Marks":[30,35,40,45,50,70,75,80,85,90]
}
 
df = pd.DataFrame(data)
 
# introduce missing value

df.loc[2, "Sleep"] = np.nan
 
# -------------------------------

# 2. HANDLE MISSING VALUES

# -------------------------------

df["Sleep"] = df["Sleep"].fillna(df["Sleep"].mean())
 
# -------------------------------

# 3. FEATURES & TARGET

# -------------------------------

X = df[["Hours","Sleep","Stress"]]

y = df["Marks"]
 
# -------------------------------
# 4. TRAIN TEST SPLIT
# -------------------------------

from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
# -------------------------------
# 5. MODEL
# -------------------------------

from sklearn.tree import DecisionTreeRegressor
 
model = DecisionTreeRegressor(

    max_depth=3,

    min_samples_split=2,

    min_samples_leaf=1

)
 
# -------------------------------
# 6. TRAIN
# -------------------------------

model.fit(X_train, y_train)
 
# -------------------------------

# 7. PREDICT

# -------------------------------

y_pred = model.predict(X_test)
 
# -------------------------------

# 8. EVALUATION

# -------------------------------

from sklearn.metrics import mean_squared_error
 
print("MSE:", mean_squared_error(y_test, y_pred))
 
# -------------------------------

# 9. TEST NEW DATA

# -------------------------------

new_data = [[6, 5, 7]]  # Hours, Sleep, Stress
 
print("Predicted Marks:", model.predict(new_data))
 