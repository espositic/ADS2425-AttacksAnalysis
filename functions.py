import pandas as pd
import collections
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import math
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import model_selection
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score

def load(path):
    return pd.read_csv(path)

def preElaborationData(df, columns):
    for c in columns:
        print('Column name: ', c, '\n')
        print(df[c].describe(), '\n')

def removeColumns(df, columns):
    removedColumns = []
    shape = df.shape
    for c in columns:
        if df[c].min() == df[c].max():
            removedColumns.append(c)
    df.drop(columns=removedColumns, inplace=True)
    print('Removed columns: ', removedColumns)
    print('Dim before the removal:', shape)
    print('Dim after the removal:', df.shape, '\n')
    return df, removedColumns

def preElaborationClass(data, c):
    value_counts = data[c].value_counts()
    plt.figure(figsize=(10, 5))
    plt.bar(value_counts.index, value_counts.values, color='skyblue')
    plt.xlabel(c)
    plt.ylabel("Frequenza")
    plt.title(f"Distribuzione della variabile '{c}'")
    plt.xticks(rotation=45)  # Ruota le etichette se sono lunghe
    plt.show()

def mutualInfoRank(data,independentList,label):
    seed = 42
    np.random.seed(seed)
    res = dict(
        zip(
            independentList,
            mutual_info_classif(data[independentList], data[label],discrete_features=False,random_state=seed)
        )
    )
    sorted_x = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_x

def topFeatureSelect(data,N):
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    top_features = [attribute for attribute, value in sorted_data[:N]]
    return top_features

def infogain(feature, label):

    value_counts = label.value_counts(normalize=True)
    original_entropy = -sum(value_counts * value_counts.apply(lambda x: math.log(x, 5)))
    feature_values = feature.unique()
    weighted_entropy = 0

    for value in feature_values:
        subset_label = label[feature == value]
        value_counts = subset_label.value_counts(normalize=True)
        subset_label_entropy = -sum(value_counts * value_counts.apply(lambda x: math.log(x, 5)))
        weighted_entropy += (len(subset_label) / len(label)) * subset_label_entropy

    return original_entropy - weighted_entropy

def giClassif(data,label):
    cols = list(data.columns.values)
    info=[]
    for c in cols:
        info.append(infogain(data[c],label))
    return info

def giRank(data,independentList,label):
    res = dict(
        zip(independentList, giClassif(data[independentList], data[label]))
    )
    sorted_x = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
    print(res)
    print(sorted_x)
    return sorted_x

def pca(data):
    pca = PCA().fit(data)
    pca_list = []
    for c in range(len(data.columns.values)):
        v="pc_"+str(c+1)
        pca_list.append(v)
    return pca,pca_list

def applyPCA(df, pca_model, pc_names):
    pcs = pca_model.transform(df)
    pc_df = pd.DataFrame(data=pcs, columns=pc_names)
    return pc_df

def decisionTreeLearner(X, y, criterion, min_samples_split=500):
    tree = DecisionTreeClassifier(criterion=criterion,min_samples_split=min_samples_split)
    tree.fit(X, y)
    return tree

def showTree(tree):
    plt.figure(figsize=(40, 30))
    plot_tree(tree, filled=True, fontsize=8, proportion=True)
    plt.show()
    nNodes = tree.tree_.node_count
    nLeaves = tree.tree_.n_leaves
    print("\nThe tree has ", nNodes, "nodes and ", nLeaves, " leaves.\n")

def decisionTreeF1(XTest, YTest, T):
    yPred = T.predict(XTest)
    f1score = f1_score(YTest, yPred, average='weighted')
    return f1score


def stratifiedKfold(X, y, folds, seed):
    skf = model_selection.StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)

    # empty lists declaration
    xTrainList = []
    xTestList = []
    yTrainList = []
    yTestList = []

    # looping over split
    for trainIndex, testIndex in skf.split(X, y):
        print("TRAIN:", trainIndex, "TEST:", testIndex)
        xTrainList.append(X.iloc[trainIndex])
        xTestList.append(X.iloc[testIndex])
        yTrainList.append(y.iloc[trainIndex])
        yTestList.append(y.iloc[testIndex])
    return xTrainList, xTestList, yTrainList, yTestList

import numpy as np

def determineDecisionTreeFoldConfiguration(xTrainList, xTestList, yTrainList, yTestList, feature_ranking):
    criterionList = ['entropy', 'gini']
    bestCriterion = ''
    bestF1score = 0
    bestNumFeatures = 0
    counter = 0

    for criterion in criterionList:
        for num_features in range(1, len(feature_ranking) + 1):  # Start from 1 feature to all features
            f1Values = []

            for x_train, y_train, x_test, y_test in zip(xTrainList, yTrainList, xTestList, yTestList):
                counter += 1
                selected_features = feature_ranking[:num_features]
                print(num_features)
                print(selected_features)
                x_train_selected = x_train[selected_features]
                x_test_selected = x_test[selected_features]

                t = decisionTreeLearner(x_train_selected, y_train, criterion)
                f1score = decisionTreeF1(x_test_selected, y_test, t)
                f1Values.append(f1score)

                print('***************************')
                print('Iteration:', counter)
                print('Criterion:', criterion)
                print(f'Number of features: {num_features}')
                print('f1score:', f1score)
                print('f1Values:', f1Values)

            avgF1 = np.mean(f1Values)
            print('avgF1:', avgF1)

            if avgF1 > bestF1score:
                bestF1score = avgF1
                bestCriterion = criterion
                bestNumFeatures = num_features

    print("bestCriterion ", bestCriterion)
    print("bestNumFeatures ", bestNumFeatures)
    print("bestF1score ", bestF1score)
    return bestCriterion, bestNumFeatures, bestF1score


def computeConfusionMatrix(yTest, yPred, modelName):

    cm = confusion_matrix(yTest, yPred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix for ' + modelName)
    plt.show()


def stratified_random_sampling(X, y, n_splits, train_size, num_features):
    sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=42)
    samples = []
    selected_features = []

    for train_index, test_index in sss.split(X, y):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        selected_columns = np.random.choice(X_train.columns, num_features, replace=False)
        selected_features.append(selected_columns)
        X_train_randomized = X_train[selected_columns]
        samples.append((X_train_randomized, y_train))

    return samples, selected_features


def train_svm(samples):
    weighted_fscore = make_scorer(f1_score, average='weighted')
    best_models = []

    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']
                  }

    for i, (X_sample, y_sample) in enumerate(samples):
        print(f"Addestramento modello SVM {i + 1}...")
        X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
        svm_model = SVC()
        grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring=weighted_fscore, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models.append(grid_search.best_estimator_)

        print(f"Modello {i + 1} ottimizzato. Migliori parametri: {grid_search.best_params_}")
        print(f"F-score ponderato sul set di validazione: {grid_search.best_score_}")
        print("-" * 40)

    return best_models

def majority_voting(models, X, selected_features):
    predictions = []

    for model, features in zip(models, selected_features):
        X_selected = X[features]  # Seleziona solo le feature usate durante il training
        predictions.append(model.predict(X_selected))

    predictions = np.array(predictions).T  # Trasponi per ottenere le predizioni per ciascun campione

    # Calcola la classe più frequente per ogni riga (campione)
    final_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)

    return final_predictions