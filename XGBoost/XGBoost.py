import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib import pyplot
import pickle
import matplotlib.pyplot as plt


# Function to load data as per required features
def loaddata(features, labels, requirement):
    data = pd.read_csv(features)
    labels = pd.read_csv(labels)

    file = open(requirement, 'r')
    lines = file.readlines()
    features = []
    for line in lines:
        features += [line[:-1]]
    features += [lines[-1]]

    features += ['key']

    # removing duplicated features
    data = data[features].loc[:, ~data[features].columns.duplicated()]

    return data, labels


# Function to sanitize data
def cleandata(data, labels):
    # removing the duplicat data points
    # there are no missing values
    data.drop_duplicates('key', inplace=True)

    # merging features and labels, keywise
    data = pd.merge(left=data, right=labels)

    # dropping key feature as it is not required for model
    data.drop('key', axis=1, inplace=True)

    # dropping correlated features with greater then 0.8 correlation
    def correlation(dataset, threshold):
        col_corr = set()
        corr_matrix = dataset.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i]
                    col_corr.add(colname)
        return col_corr

    corr_feat = list(correlation(data.drop(['label'], axis=1), 0.8))
    # saving correlated features for removing while prediction
    file_name = "corr_feat.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(corr_feat, open_file)
    open_file.close()

    return data.drop(corr_feat, axis=1)


# Funtion to train and output the requirements
def train(data):
    # train test split
    x = data.drop(['label'], axis=1)
    y = data['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # hyper parameter tuning via RandomizedSearchCV cross validation
    #log_grid = {
        #    'n_estimators': [100, 1000],
        # 'max_depth': [35, 45, 55]}
        #    'gamma': [0, 10, 100],
    #    'eta': [0.01, 0.1, 1]}
    #    'lambda': [0, 10, 100],
    #    'alpha': [0, 10, 100]
    # }
    #clf_xgb = GridSearchCV(
     #   estimator=xgb.XGBClassifier(),
     #   param_grid=log_grid,
    #    scoring=make_scorer(accuracy_score),
     #   cv=5,
    #    verbose=0
    #)
    #clf_xgb.fit(x_train, y_train)
    #print(clf_xgb.best_params_)
    # fitting model with best params
    # n_estimators = clf_xgb.best_params_['n_estimators']
    #max_depth = clf_xgb.best_params_['max_depth']
    # learning_rate = clf_xgb.best_params_['learning_rate']
    # booster = clf_xgb.best_params_['booster']
    # gamma = clf_xgb.best_params_['gamma']
    # max_delta_step = clf_xgb.best_params_['max_delta_step']
    # eta = clf_xgb.best_params_['eta']
    # lambdav = clf_xgb.best_params_['lambda']
    # alpha = clf_xgb.best_params_['alpha']
    clf_xgb_best = xgb.XGBClassifier(reg_lambda=0,
                                     n_estimators=100,
    #                                 learning_rate=learning_rate,
                                     max_depth=35,
    #                                 booster=booster,
                                     gamma=10,
    #                                 max_delta_step=max_delta_step,
                                     eta=0.1,
                                     alpha=0).fit(x_train, y_train)

    # saving the model
    filename = 'XGBoost/XGBoostModel.sav'
    pickle.dump(clf_xgb_best, open(filename, 'wb'))

    # ROC Curve
    rocauc(clf_xgb_best, x_test, y_test)

    # Feature Importance
    feat_imp(clf_xgb_best, data)

    return


# Function to plot ROC Curve
def rocauc(model, x_test, y_test):
    pred_prob = model.predict_proba(x_test)
    fpr, tpr, thresh = roc_curve(y_test, pred_prob[:, 1], pos_label=1)

    random_probs = [0 for _ in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

    val = roc_auc_score(y_test, model.predict(x_test))
    lab = 'XGBoost (roc_auc_score = ' + str(val)[:5] + ')'

    plt.figure(figsize=(5, 5))
    plt.style.use('seaborn')
    plt.plot(fpr, tpr, linestyle='--', color='orange', label=lab)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    plt.title('XGBoost ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend()
    plt.savefig('XGBoost/XGBoost', dpi=300)

    return


# Function for finding feature importance
def feat_imp(model, data):
    data1 = {}
    for i in range(len(data.columns) - 1):
        data1[data.columns[i]] = model.feature_importances_[i]
    data1 = dict(sorted(((value, key) for (key, value) in data1.items()), reverse=True))

    features = list(data1.keys())
    importance_value = list(data1.values())

    pyplot.figure(figsize=(20, 10))
    pyplot.bar(importance_value, features)
    plt.xticks(rotation=80, fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Feature Name", fontsize=15)
    plt.ylabel("Importance Value", fontsize=15)
    plt.title("XGBoost Feature Importance", fontsize=20)
    plt.savefig('XGBoost/XGBoostFeatureImportance', dpi=300)

    return


# Overall training script function
def start(inputfeatures, inputlabels, inputrequirements='./features.txt'):
    data, labels = loaddata(inputfeatures, inputlabels, inputrequirements)

    data_f = cleandata(data, labels)

    train(data_f)

    return


start('./features.csv', './labels.csv')
