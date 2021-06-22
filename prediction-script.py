import pandas as pd
import pickle


def pred_data(file, features):
    file1 = open(features, 'r')
    lines = file1.readlines()
    features = []
    for line in lines:
        features += [line[:-1]]
    features += [lines[-1]]

    data = pd.read_csv(file)

    # removing duplicated features
    data = data[features].loc[:, ~data[features].columns.duplicated()]

    open_file = open('corr_feat.pkl', "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()

    for feat in loaded_list:
        if feat in data:
            data.drop([feat], axis=1, inplace=True)

    return data


def LogisticRegression(data):
    # load the model trained
    loaded_model = pickle.load(open('LogisticRegression/LogisticRegressionModel.sav', 'rb'))

    return loaded_model.predict_proba(data)


def RandomForest(data):
    # load the model trained
    loaded_model = pickle.load(open('RandomForest/RandomForestModel.sav', 'rb'))

    return loaded_model.predict_proba(data)


def XGBoost(data):
    # load the model trained
    loaded_model = pickle.load(open('XGBoost/XGBoostModel.sav', 'rb'))

    return loaded_model.predict_proba(data)


def predictions(data, model='all', features='features.txt'):
    data = pred_data(data, features)
    preds = {}

    if model == 'LogisticRegression':
        return LogisticRegression(data)
    elif model == 'RandomForest':
        return RandomForest(data)
    elif model == 'XGBoost':
        return XGBoost(data)
    else:
        preds['LogisticRegression'] = LogisticRegression(data)
        preds['RandomForest'] = RandomForest(data)
        preds['XGBoost'] = XGBoost(data)

    return preds

# running whole script
print(predictions('predict.csv'))
