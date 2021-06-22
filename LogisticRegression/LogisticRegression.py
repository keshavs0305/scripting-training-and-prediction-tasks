import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
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

    #    #removing duplicated features
    data = data[features].loc[:, ~data[features].columns.duplicated()]

    return data, labels


# Function to sanitize data
def cleandata(data, labels):
    #    #removing the duplicat data points
    #    #there are no missing values
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
    log_grid = {
        'tol': [0.0001, 0.01],
        'C': [0.0001, 0.01, 0.1],
        'max_iter': [100, 1000]
    }
    clf_log_cv = RandomizedSearchCV(
        estimator=LogisticRegression(verbose=0),
        param_distributions=log_grid,
        scoring=make_scorer(accuracy_score),
        cv=5,
        verbose=0
    )
    clf_log_cv.fit(x_train, y_train)

    # fitting model with best params
    tol = clf_log_cv.best_params_['tol']
    max_iter = clf_log_cv.best_params_['max_iter']
    c = clf_log_cv.best_params_['C']
    clf_log_best = LogisticRegression(tol=tol, max_iter=max_iter, C=c).fit(x_train, y_train)

    # saving the model
    filename = 'LogisticRegression/LogisticRegressionModel.sav'
    pickle.dump(clf_log_best, open(filename, 'wb'))

    # ROC Curve
    rocauc(clf_log_best, x_test, y_test)

    # Feature Importance
    feat_imp(clf_log_best, data)

    return


# Function to plot ROC Curve
def rocauc(model, x_test, y_test):
    pred_prob = model.predict_proba(x_test)
    fpr, tpr, thresh = roc_curve(y_test, pred_prob[:, 1], pos_label=1)

    random_probs = [0 for _ in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

    val = roc_auc_score(y_test, model.predict(x_test))
    lab = 'Logistic Regression (roc_auc_score = ' + str(val)[:5] + ')'

    plt.figure(figsize=(5, 5))
    plt.style.use('seaborn')
    plt.plot(fpr, tpr, linestyle='--', color='orange', label=lab)
    plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
    plt.title('Logistic Regression ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend()
    plt.savefig('LogisticRegression/LogisticRegressionROC', dpi=300)

    return


# Function for finding feature importance
def feat_imp(model, data):
    data1 = {}
    for i in range(len(data.columns) - 1):
        data1[data.columns[i]] = model.coef_[0][i]
    data1 = dict(sorted(((value, key) for (key, value) in data1.items()), reverse=True))

    features = list(data1.keys())
    importance_value = list(data1.values())

    pyplot.figure(figsize=(20, 10))
    pyplot.bar(importance_value, features)
    plt.xticks(rotation=80, fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Feature Name", fontsize=15)
    plt.ylabel("Importance Value", fontsize=15)
    plt.title("Logistic Regression Feature Importance", fontsize=20)
    plt.savefig('LogisticRegression/LogisticRegressionFeatureImportance', dpi=300)

    return


# Overall training script function
def start(inputfeatures, inputlabels, inputrequirements='./features.txt'):
    data, labels = loaddata(inputfeatures, inputlabels, inputrequirements)

    data_f = cleandata(data, labels)

    train(data_f)

    return

# running whole script
start('./features.csv', './labels.csv')
