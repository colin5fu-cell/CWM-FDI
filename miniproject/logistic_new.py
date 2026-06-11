import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer

data = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")

x = data.drop("Diabetes_012", axis=1)
y = data["Diabetes_012"]

continuous = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 1)

scaler = StandardScaler()

x_train[continuous] = scaler.fit_transform(x_train[continuous])
x_test[continuous] = scaler.transform(x_test[continuous])

model = LogisticRegression()

model.fit(x_train, y_train)

y_predicted = model.predict(x_test)

score = accuracy_score(y_predicted, y_test)

print("accuracy score is ", score)
