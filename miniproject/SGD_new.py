import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

"""script that reads data and runs the SGD classifier"""

data = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

x = data.drop("Diabetes_binary", axis=1)
y = data["Diabetes_binary"]

continuous = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state = 1)

scaler = StandardScaler()

x_train[continuous] = scaler.fit_transform(x_train[continuous])
x_test[continuous] = scaler.transform(x_test[continuous])

model = SGDClassifier(class_weight = 'balanced')

model.fit(x_train, y_train)

y_predicted = model.predict(x_test)

report = classification_report(y_test, y_predicted, output_dict = True)

print(report)
