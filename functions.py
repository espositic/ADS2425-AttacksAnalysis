import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import math
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, make_scorer, f1_score
from sklearn import model_selection
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import re


def load(path):
    """
    Carica un file CSV e restituisce un DataFrame pandas.

    Args:
        path (str): Percorso del file CSV da leggere.

    Returns:
        pd.DataFrame: DataFrame contenente i dati caricati dal file.
    """
    return pd.read_csv(path)


def preElaborationData(df, columns):
    """
    Stampa statistiche descrittive per un elenco di colonne di un DataFrame.

    Args:
        df (pd.DataFrame): Il DataFrame da analizzare.
        columns (list): Lista di nomi di colonne per cui mostrare le statistiche.

    Returns:
        None
    """
    for c in columns:
        print('Column name: ', c, '\n')
        print(df[c].describe(), '\n')


def removeColumns(df, columns):
    """
    Rimuove dal DataFrame le colonne il cui valore minimo è uguale al valore massimo (colonne costanti).

    Args:
        df (pd.DataFrame): Il DataFrame da cui rimuovere le colonne.
        columns (list): Lista di nomi di colonne da controllare.

    Returns:
        tuple:
            - pd.DataFrame: DataFrame con le colonne rimosse.
            - list: Lista delle colonne rimosse.
    """
    removedColumns = []

    for c in columns:
        if df[c].min() == df[c].max():
            removedColumns.append(c)

    df.drop(columns=removedColumns, inplace=True)
    return df, removedColumns


def preElaboration(df, class_column, output_dir="boxplot"):
    """
    Genera e salva i boxplot per le colonne numeriche di un DataFrame, raggruppate per la colonna di classe.

    Args:
        df (pd.DataFrame): Il DataFrame contenente i dati.
        class_column (str): Nome della colonna utilizzata per il raggruppamento nel boxplot.
        output_dir (str, optional): Directory dove salvare i boxplot. Default è "boxplot".

    Returns:
        None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    numerical_cols = df.select_dtypes(include=['number']).columns.str.strip()

    for col in numerical_cols:
        safe_col_name = re.sub(r'[^a-zA-Z0-9]', '_', col)
        ax = df.boxplot(column=col, grid=False, by=class_column)
        ax.set_title(f'Boxplot of {col}')
        fig = ax.get_figure()
        fig.savefig(os.path.join(output_dir, f'{safe_col_name}_boxplot.png'))
        fig.clf()


def preElaborationClass(data, c):
    """
    Visualizza un grafico a barre della distribuzione dei valori per una colonna categorica.

    Args:
        data (pd.DataFrame): Il DataFrame contenente i dati.
        c (str): Nome della colonna da analizzare.

    Returns:
        None
    """
    value_counts = data[c].value_counts()
    plt.figure(figsize=(10, 5))
    plt.bar(value_counts.index, value_counts.values)
    plt.xlabel(c)
    plt.ylabel("Frequenza")
    plt.title(f"Distribuzione della variabile '{c}'")
    plt.show()


def mutualInfoRank(data, independentList, label):
    """
    Calcola e ordina l'importanza delle feature rispetto alla variabile target usando l'informazione mutua.

    Args:
        data (pd.DataFrame): DataFrame contenente i dati.
        independent_list (list): Lista delle colonne/features indipendenti da valutare.
        label (str): Nome della colonna target (variabile dipendente).

    Returns:
        list of tuples: Lista di tuple (feature, punteggio) ordinata dal punteggio più alto al più basso.
    """
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


def topFeatureSelect(data, N):
    """
    Seleziona le prime N feature con i punteggi più alti da una lista ordinata di tuple.

    Args:
        data (list of tuples): Lista di tuple (feature, punteggio) ordinata o non ordinata.
        N (int): Numero di feature da selezionare.

    Returns:
        list: Lista delle prime N feature selezionate.
    """
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    top_features = [attribute for attribute, value in sorted_data[:N]]
    return top_features


def infogain(feature, label):
    """
    Calcola il guadagno informativo (information gain) di una feature rispetto alla variabile target.

    Args:
        feature (pd.Series): La feature da valutare.
        label (pd.Series): La variabile target.

    Returns:
        float: Il valore dell'information gain.
    """
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
    """
    Calcola il guadagno informativo per ogni colonna di un DataFrame rispetto alla variabile target.

    Args:
        data (pd.DataFrame): DataFrame contenente le feature da valutare.
        label (pd.Series): Variabile target.

    Returns:
        list: Lista dei valori di guadagno informativo per ogni feature.
    """
    cols = data.columns.to_list()
    info=[]

    for c in cols:
        info.append(infogain(data[c],label))

    return info


def giRank(data,independentList,label):
    """
    Calcola e ordina il guadagno informativo delle feature rispetto alla variabile target.

    Args:
        data (pd.DataFrame): DataFrame contenente le feature e la variabile target.
        independentList (list): Lista delle colonne/features indipendenti da valutare.
        label (str): Nome della colonna target (variabile dipendente).

    Returns:
        list of tuples: Lista di tuple (feature, guadagno informativo) ordinata dal valore più alto al più basso.
    """
    res = dict(
        zip(independentList, giClassif(data[independentList], data[label]))
    )
    sorted_x = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
    print(res)
    print(sorted_x)
    return sorted_x


def pca(data):
    """
    Esegue l'analisi delle componenti principali (PCA) sui dati forniti.

    Args:
        data (pd.DataFrame): DataFrame contenente i dati numerici da analizzare.

    Returns:
        tuple:
            - PCA: Oggetto PCA addestrato sui dati.
            - list: Lista di nomi delle componenti principali generate (es. ["pc_1", "pc_2", ...]).
    """
    pca = PCA().fit(data)
    pca_list = []

    for c in range(len(data.columns.values)):
        v="pc_"+str(c+1)
        pca_list.append(v)

    return pca,pca_list


def applyPCA(df, pca_model, pc_names):
    """
    Applica il modello PCA ai dati e restituisce un DataFrame con le componenti principali.

    Args:
        df (pd.DataFrame): DataFrame contenente i dati da trasformare.
        pca_model (PCA): Modello PCA addestrato.
        pc_names (list): Lista dei nomi delle componenti principali.

    Returns:
        pd.DataFrame: DataFrame contenente le componenti principali come colonne.
    """
    pcs = pca_model.transform(df)
    pc_df = pd.DataFrame(data=pcs, columns=pc_names)
    return pc_df


def decisionTreeLearner(X, y, criterion, min_samples_split=500):
    """
    Addestra un classificatore ad albero decisionale con i parametri specificati.

    Args:
        X (pd.DataFrame or np.ndarray): Feature di input per l'addestramento.
        y (pd.Series or np.ndarray): Variabile target.
        criterion (str): Criterio di divisione ("gini" o "entropy").
        min_samples_split (int, optional): Numero minimo di campioni richiesti per dividere un nodo. Default è 500.

    Returns:
        DecisionTreeClassifier: Modello addestrato.
    """
    tree = DecisionTreeClassifier(criterion=criterion,min_samples_split=min_samples_split)
    tree.fit(X, y)
    return tree


def showTree(tree):
    """
    Visualizza l'albero decisionale e stampa il numero di nodi e foglie.

    Args:
        tree (DecisionTreeClassifier): Modello ad albero decisionale addestrato.

    Returns:
        None
    """
    plt.figure(figsize=(40, 30))
    plot_tree(tree, filled=True, fontsize=8, proportion=True)
    plt.show()
    nNodes = tree.tree_.node_count
    nLeaves = tree.tree_.n_leaves
    print("\nL'albero ha ", nNodes, "nodi e ", nLeaves, " foglie.\n")


def decisionTreeF1(XTest, YTest, T):
    """
    Calcola il punteggio F1 (weighted) per un modello decision tree su dati di test.

    Args:
        XTest (pd.DataFrame or np.ndarray): Feature del set di test.
        YTest (pd.Series or np.ndarray): Target reale del set di test.
        T (DecisionTreeClassifier): Modello addestrato da valutare.

    Returns:
        float: Punteggio F1 weighted ottenuto dal modello sul set di test.
    """
    yPred = T.predict(XTest)
    f1score = f1_score(YTest, yPred, average='weighted')
    return f1score


def stratifiedKfold(X, y, folds, seed):
    """
    Esegue una suddivisione stratificata in K-fold per dati e target forniti.

    Args:
        X (pd.DataFrame): Feature del dataset.
        y (pd.Series): Variabile target.
        folds (int): Numero di fold per la cross-validation.
        seed (int): Random seed per la riproducibilità.

    Returns:
        tuple: Quattro liste contenenti:
            - xTrainList (list of pd.DataFrame): Liste di dati di training per ogni fold.
            - xTestList (list of pd.DataFrame): Liste di dati di test per ogni fold.
            - yTrainList (list of pd.Series): Liste di target di training per ogni fold.
            - yTestList (list of pd.Series): Liste di target di test per ogni fold.
    """
    skf = model_selection.StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)

    xTrainList = []
    xTestList = []
    yTrainList = []
    yTestList = []

    for trainIndex, testIndex in skf.split(X, y):
        xTrainList.append(X.iloc[trainIndex])
        xTestList.append(X.iloc[testIndex])
        yTrainList.append(y.iloc[trainIndex])
        yTestList.append(y.iloc[testIndex])
    return xTrainList, xTestList, yTrainList, yTestList


def determineDecisionTreeFoldConfiguration(xTrainList, xTestList, yTrainList, yTestList, feature_ranking):
    """
    Determina la configurazione migliore per un albero decisionale valutando
    diversi criteri e il numero di feature selezionate tramite cross-validation stratificata.

    Args:
        xTrainList (list of pd.DataFrame): Liste di dati di training per ogni fold.
        xTestList (list of pd.DataFrame): Liste di dati di test per ogni fold.
        yTrainList (list of pd.Series): Liste di target di training per ogni fold.
        yTestList (list of pd.Series): Liste di target di test per ogni fold.
        feature_ranking (list): Lista ordinata delle feature secondo il loro ranking di importanza.

    Returns:
        tuple:
            - bestCriterion (str): Il criterio ('entropy' o 'gini') che ha prodotto il miglior punteggio F1 medio.
            - bestNumFeatures (int): Il numero ottimale di feature da usare.
            - bestF1score (float): Il miglior punteggio F1 medio ottenuto.
    """
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
    """
    Calcola e visualizza la matrice di confusione e stampa il classification report per un modello.

    Args:
        yTest (array-like): Target reale.
        yPred (array-like): Predizioni del modello.
        modelName (str): Nome del modello, usato nel titolo della figura e stampa.

    Returns:
        None
    """
    cm = confusion_matrix(yTest, yPred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Matrice di confusione per ' + modelName)
    plt.show()
    print(classification_report(yTest, yPred))


def scale_train_test(X_train, testset):
    """
    Standardizza (Z-score scaling) il training set e il test set
    restituendo due DataFrame Pandas con colonne e indici preservati.

    Args:
        - X_train: DataFrame training features
        - testset: DataFrame test set completo (include anche target)

    Returns:
        - X_train_scaled_df: DataFrame con training features scalate
        - testset_scaled_df: DataFrame con test features scalate
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    testset_scaled = scaler.transform(testset[X_train.columns])

    X_train_scaled_df = pd.DataFrame(X_scaled, columns=X_train.columns, index=X_train.index)
    testset_scaled_df = pd.DataFrame(testset_scaled, columns=X_train.columns, index=testset.index)

    return X_train_scaled_df, testset_scaled_df


def stratified_random_sampling(X, y, n_splits, train_size, num_features):
    """
    Esegue un campionamento stratificato casuale con selezione randomica di feature.

    Args:
        X (pd.DataFrame): Feature del dataset.
        y (pd.Series): Variabile target.
        n_splits (int): Numero di split da generare.
        train_size (float): Percentuale del dataset da usare come training in ogni split.
        num_features (int): Numero di feature da selezionare casualmente per ogni split.

    Returns:
        tuple:
            - samples (list): Lista di tuple (X_train_randomized, y_train) per ogni split.
            - selected_features (list): Lista di array contenenti i nomi delle feature selezionate in ogni split.
    """
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
    """
    Addestra modelli SVM su più campioni, effettuando una ricerca esaustiva di iperparametri con GridSearchCV.

    Args:
        samples (list): Lista di tuple (X_sample, y_sample), dati e target per ogni campione.

    Returns:
        list: Lista di modelli SVM ottimizzati per ogni campione.
    """
    weighted_fscore = make_scorer(f1_score, average='weighted')
    best_models = []

    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf']}

    for i, (X_sample, y_sample) in enumerate(samples):
        print(f"Addestramento modello SVM {i + 1}...")
        X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
        svm_model = SVC()
        grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring=weighted_fscore, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models.append(grid_search.best_estimator_)
        print(f"Modello {i + 1}. Migliori parametri: {grid_search.best_params_}")
        print(f"F-score pesato sul validazione set: {grid_search.best_score_}")
        print("-" * 40)

    return best_models

def majority_voting(models, X, selected_features):
    """
    Effettua una predizione aggregata tramite voto di maggioranza da più modelli.

    Args:
        models (list): Lista di modelli addestrati.
        X (pd.DataFrame): Dataset sul quale effettuare le predizioni.
        selected_features (list): Lista di liste/array con le feature usate da ogni modello.

    Returns:
        np.ndarray: Array delle predizioni finali ottenute tramite voto di maggioranza.
    """
    predictions = []

    for model, features in zip(models, selected_features):
        X_selected = X[features]
        predictions.append(model.predict(X_selected))

    predictions = np.array(predictions).T
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)