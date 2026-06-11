import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("diabetes.csv")

x = data.drop("Outcome", axis=1)
y = data["Outcome"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 1)
 
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model = BaggingClassifier()

model.fit(x_train_scaled, y_train)

y_predicted = model.predict(x_test_scaled)

score = accuracy_score(y_predicted, y_test)

print("accuracy score is ", score)
