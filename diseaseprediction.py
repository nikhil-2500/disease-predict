import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import csv
import numpy as np
import pandas as pd
import os

# read in the CSV file and perform calculations to create the dictionary
disease_symptom_prob = {}
with open('top_5_Symptoms.csv') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)  # skip the header row
    for row in reader:
        disease = row[0]
        for col in row[1:]:
            symptom, value = col.split(': ')
            symptom_prob = float(value.strip('%'))

            if disease not in disease_symptom_prob:
                disease_symptom_prob[disease] = {}

            disease_symptom_prob[disease][symptom] = symptom_prob

# Load the trained hybrid model
hybrid_model = joblib.load('hybrid_model.joblib')
data = pd.read_csv(os.path.join("templates", "Testing.csv"))
df = pd.DataFrame(data)
indices = [i for i in range(132)]
symptoms = df.columns.values[:-1]
dictionary = dict(zip(symptoms, indices))


def dosomething(symptom):
    user_input_symptoms = symptom
    user_input_label = [0 for i in range(132)]
    for i in user_input_symptoms:
        idx = dictionary[i]
        user_input_label[idx] = 1

    user_input_label = np.array(user_input_label)
    user_input_label = user_input_label.reshape((-1, 1)).transpose()

    # Prepare the input data
    custom_input = user_input_label
    # to get prediction on individual classifiers in the model
    cn = []
    class_index = hybrid_model.named_estimators_['dt'].predict(custom_input)
    class_name = hybrid_model.classes_[int(class_index)]
    cn.append(class_name)
    class_index = hybrid_model.named_estimators_['knn'].predict(custom_input)
    class_name = hybrid_model.classes_[int(class_index)]
    cn.append(class_name)
    class_index = hybrid_model.named_estimators_['rf'].predict(custom_input)
    class_name = hybrid_model.classes_[int(class_index)]
    cn.append(class_name)
    class_index = hybrid_model.named_estimators_['nb'].predict(custom_input)
    class_name = hybrid_model.classes_[int(class_index)]
    cn.append(class_name)
    class_name = hybrid_model.predict(custom_input)[0]
    cn.append(class_name)

    cn.sort()
    name = cn[0]
    gt = 0
    temp = 0
    for a in range(0, cn.__len__()):
        # if a==0 or cn[a]!=cn[a-1]:
        temp = 0
        for b in symptom:
            try:
                temp += disease_symptom_prob[cn[a]][b]
                # print(cn[a], b, disease_symptom_prob[cn[a]][b], temp)
            except:
                pass
        if temp > gt:
            if gt > 100:  # for a secondary disease that also is present
                gt = temp
                name = cn[a]+" & "+name
            else:
                gt = temp
                name = cn[a]
    print("The final disease prediction is:", name)
    return (name)

# # old
#     user_input_symptoms = symptom
#     user_input_label = [0 for i in range(132)]
#     for i in user_input_symptoms:
#         idx = dictionary[i]
#         user_input_label[idx] = 1

#     user_input_label = np.array(user_input_label)
#     user_input_label = user_input_label.reshape((-1, 1)).transpose()
#     return (dt.predict(user_input_label))
