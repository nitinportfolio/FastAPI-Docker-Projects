import json
from xgboost import XGBClassifier


def make_prediction(x):
    # Loading the model
    loaded_model = XGBClassifier()
    loaded_model.load_model("main_model.model")
    predictions_out = loaded_model.predict(x)
    print(predictions_out)

    # Make predictions
    dict_out = {}
    for count, value in enumerate(predictions_out):
        dict_out[count] = float(value)

    # loading json for decoding & decode output
    with open("encoder.json") as json_file:
        data = json.load(json_file)

    # Returns the species name
    return data[str(int(dict_out[0]))]
