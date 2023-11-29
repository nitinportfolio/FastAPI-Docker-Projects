import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier
import csv
import json


def make_model_save():
    # Getting iris data
    iris_data = pd.read_csv(
        ".\iris.data",
        header=None,
        names=["Sepal Length", "Sepal Width",
               "Petal Length", "Petal Width",
               "Species"]
    )

    # Processing data
    label_encoder = LabelEncoder()
    iris_data["Species_Encoded"] = label_encoder.fit_transform(iris_data["Species"])

    # Saving Label encoded data to a new file & json
    iris_data.to_csv("encoded_data.csv")
    options_title = iris_data["Species"].unique()
    dict_encoder = {}
    for item in options_title:
        dict_encoder[str(iris_data[iris_data["Species"] == item].iloc[0]["Species_Encoded"])] = item

    with open("encoder.json", "w") as write_file:
        json.dump(dict_encoder, write_file, indent=4)

    # Make x and y data
    y = iris_data["Species_Encoded"].copy()
    X = iris_data.drop(["Species", "Species_Encoded"], axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8)

    # Building & Training Model & Saving it
    model = XGBClassifier()
    model.fit(X_train, y_train)
    model.save_model("main_model.model")

    # Testing Model
    predictions = model.predict(X_valid)
    print(f"MAE: {str(mean_absolute_error(predictions, y_valid))}")

    # Writing new data entered from frontend to added to the file


def add_to_data(x1, x2, x3, x4, y):
    new_row = [float(x1), float(x2), float(x3), float(x4), str(y)]

    with open("iris.data", "a") as csv_file:
        writer_object = csv.writer(csv_file)
        writer_object.writerow(new_row)
