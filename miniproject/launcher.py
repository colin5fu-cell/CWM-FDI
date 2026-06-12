import subprocess
import ast
"""script that launches four models, measures their energy consumption and justifies the best model by calculating a combined metric""" 

model1_result = subprocess.run(["python3", "logistic_new.py"], capture_output = True, text = True)
model1_energy = subprocess.run(["sudo", "turbostat", "--Summary", "--quiet", "--Joules", "--show", "Pkg_J", "python3", "logistic_new.py"], capture_output = True, text = True)

print("\nOutput Result by Logistic Regression\n", model1_result.stdout)
print("\nEnergy Consumption by Logistic Regression\nTime\n", model1_energy.stderr)

lines1 = model1_energy.stderr.splitlines()
print("\nThe carbon emission caused by the programme in South England is: ", float(lines1[2])/1000/3600*62, " grams of CO2")

model2_result = subprocess.run(["python3", "forest_new.py"], capture_output = True, text = True)
model2_energy = subprocess.run(["sudo", "turbostat", "--Summary", "--quiet", "--Joules", "--show", "Pkg_J", "python3", "forest_new.py"], capture_output = True, text = True)

print("\nOutput Result by Random Forest\n", model2_result.stdout)
print("\nEnergy Consumption by Random Forest\nTime\n", model2_energy.stderr)

lines2 = model2_energy.stderr.splitlines()
print("\nThe carbon emission caused by the programme in South England is: ", float(lines2[2])/1000/3600*62, " grams of CO2")

model3_result = subprocess.run(["python3", "gradientboost_new.py"], capture_output = True, text = True)
model3_energy = subprocess.run(["sudo", "turbostat", "--Summary", "--quiet", "--Joules", "--show", "Pkg_J", "python3", "gradientboost_new.py"], capture_output = True, text = True)

print("\nOutput Result by Gradient Boosting\n", model3_result.stdout)
print("\nEnergy Consumption by Gradient Boosting\nTime\n", model3_energy.stderr)

lines3 = model3_energy.stderr.splitlines()
print("\nThe carbon emission caused by the programme in South England is: ", float(lines3[2])/1000/3600*62, " grams of CO2")

model4_result = subprocess.run(["python3", "SGD_new.py"], capture_output = True, text = True)
model4_energy = subprocess.run(["sudo", "turbostat", "--Summary", "--quiet", "--Joules", "--show", "Pkg_J", "python3", "SGD_new.py"], capture_output = True, text = True)

print("\nOutput Result by SGD\n", model4_result.stdout)
print("\nEnergy Consumption by SGD\nTime\n", model4_energy.stderr)

lines4 = model4_energy.stderr.splitlines()
print("\nThe carbon emission caused by the programme in South England is: ", float(lines4[2])/1000/3600*62, " grams of CO2")


#extracts the energy consumption of each model
energy = [float(lines1[2]), float(lines2[2]), float(lines3[2]), float(lines4[2])]

#normalizes the four energy values to range 0-1 using (value-min.value)/max.value-min.value
max_eng = max(energy)
min_eng = min(energy)
nor_eng = [0]*4

for i in range(len(nor_eng)):
    nor_eng[i] = 1 - (energy[i]-min_eng)/(max_eng - min_eng)


#calculates the normalised f1 score
f1 = [ast.literal_eval(model1_result.stdout)["weighted avg"]["f1-score"], ast.literal_eval(model2_result.stdout)["weighted avg"]["f1-score"], ast.literal_eval(model3_result.stdout)["weighted avg"]["f1-score"], ast.literal_eval(model4_result.stdout)["weighted avg"]["f1-score"]]
max_f1 = max(f1)
min_f1 = min(f1)
nor_f1 = [0] * len(f1)

for i in range(len(f1)):
    nor_f1 = (f1[i] - min_f1)/(max_f1-min_f1)

recall1 = [
    ast.literal_eval(model1_result.stdout)["1.0"]["recall"],
    ast.literal_eval(model2_result.stdout)["1.0"]["recall"],
    ast.literal_eval(model3_result.stdout)["1.0"]["recall"],
    ast.literal_eval(model4_result.stdout)["1.0"]["recall"]
]

max_recall = max(recall1)
min_recall = min(recall1)

nor_recall = [0] * len(recall1)

for i in range(len(nor_recall)):
    nor_recall[i] = (nor_recall[i] - min_recall) / (max_recall - min_recall)

models = ["Logistic Regression", "Random Forest","Gradient Boosting", "SGD"]

metric = [
    0.25 * nor_recall[i]
    + 0.25 * nor_f1[i]
    + 0.5 * energy_score[i]
    for i in range(4)
]

for i in range(len(models)):
    print(
        f"{models[i]}: "
        f"Recall={nor_recall[i]:.4f}, "
        f"F1={nor_f1[i]:.4f}, "
        f"Energy={energy_score[i]:.4f}, "
        f"Metric={metric[i]:.4f}"
    )

