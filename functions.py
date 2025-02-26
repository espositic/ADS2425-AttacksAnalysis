import pandas as pd
import numpy as np
import os
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
import math
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import model_selection
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score, accuracy_score
import re


def load(path):
    """Carica un file CSV e restituisce un DataFrame."""
    return pd.read_csv(path)

def preElaborationData(df, columns):
    """Stampa il nome e le statistiche descrittive
    delle colonne specificate di un DataFrame."""
    for c in columns:
        print('Column name: ', c, '\n')
        print(df[c].describe(), '\n')

def removeColumns(df, columns):
    """Rimuove le colonne con valori tutti uguali e
    restituisce il DataFrame modificato e l'elenco
    delle colonne rimosse."""
    removedColumns = []
    for c in columns:
        if df[c].min() == df[c].max():
            removedColumns.append(c)
    df.drop(columns=removedColumns, inplace=True)
    return df, removedColumns

def preElaboration(df, class_column, output_dir="boxplot"):
    """
    Genera boxplot per le colonne numeriche raggruppate per la classe.

    Parametri:
    df (DataFrame): Il DataFrame da analizzare.
    class_column (str): La colonna classe.
    output_dir (str): Cartella in cui salvare i grafici (default: "boxplot").
    """
    # Create folder if it doesn't exist
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
    Visualizza la distribuzione della classe con un istogramma.

    Parametri:
    data (DataFrame): Il DataFrame contenente i dati.
    c (str): Il nome della colonna classe.
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
    Calcola e ordina il punteggio di Mutual Information
    tra le variabili indipendenti e la variabile target.

    Parametri:
    data (DataFrame): Il DataFrame contenente i dati.
    independentList (list): Lista delle variabili indipendenti.
    label (str): Nome della colonna target.

    Ritorna:
    list: Lista di tuple (feature, punteggio) ordinate in
        ordine decrescente di importanza.
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
    Seleziona le N feature con il punteggio più alto.

    Parametri:
    data (list): Lista di tuple (feature, punteggio), ordinata o da ordinare.
    N (int): Numero di caratteristiche da selezionare.

    Ritorna:
    list: Lista delle N caratteristiche più importanti.
    """
    sorted_data = sorted(data, key=lambda x: x[1], reverse=True)
    top_features = [attribute for attribute, value in sorted_data[:N]]
    return top_features

def infogain(feature, label):
    """
   Calcola Information Gain di una feature rispetto alla variabile target.

   Parametri:
   feature (Series): La colonna del DataFrame contenente la feature da analizzare.
   label (Series): La colonna del DataFrame contenente la variabile target.

   Ritorna:
   float: Il valore del guadagno di informazione.
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
    Calcola Information Gain per tutte le colonne del DataFrame rispetto
    alla variabile target.

    Parametri:
    data (DataFrame): Il DataFrame contenente le feature.
    label (Series): La colonna target rispetto alla quale calcolare il guadagno di informazione.

    Ritorna:
    list: Lista dei valori di Information Gain per ogni feature del DataFrame.
    """
    cols = list(data.columns.values)
    info=[]
    for c in cols:
        info.append(infogain(data[c],label))
    return info

def giRank(data,independentList,label):
    """
    Calcola e ordina il guadagno di informazione (Information Gain) per un
    insieme di variabili indipendenti rispetto alla variabile target.

    Parametri:
    data (DataFrame): Il DataFrame contenente i dati.
    independentList (list): Lista delle colonne indipendenti da analizzare.
    label (str): Nome della colonna target.

    Ritorna:
    list: Lista di tuple (feature, punteggio) ordinate in ordine decrescente di Information Gain.
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
    Applica l'Analisi delle Componenti Principali (PCA)
        ai dati e genera i nomi delle componenti.

    Parametri:
    data (DataFrame): Il DataFrame contenente i dati da trasformare.

    Ritorna:
    - PCA: L'oggetto PCA addestrato.
    - list: Lista con i nomi delle componenti principali.
    """
    pca = PCA().fit(data)
    pca_list = []
    for c in range(len(data.columns.values)):
        v="pc_"+str(c+1)
        pca_list.append(v)
    return pca,pca_list

def applyPCA(df, pca_model, pc_names):
    """
    Applica un modello PCA a un DataFrame e restituisce le componenti principali.

    Parametri:
    df (DataFrame): Il DataFrame su cui applicare la PCA.
    pca_model (PCA): Il modello PCA già addestrato.
    pc_names (list): Lista con i nomi delle componenti principali.

    Ritorna:
    DataFrame: Un nuovo DataFrame contenente le componenti principali.
    """
    pcs = pca_model.transform(df)
    pc_df = pd.DataFrame(data=pcs, columns=pc_names)
    return pc_df

def decisionTreeLearner(X, y, criterion, min_samples_split=500):
    """
    Crea e addestra un albero di decisione per la classificazione.

    Parametri:
    X (DataFrame o array-like): Le feature indipendenti.
    y (Series o array-like): La variabile target.
    criterion (str): Il criterio di suddivisione ('gini' o 'entropy').
    min_samples_split (int, opzionale): Numero minimo di campioni richiesti per dividere un nodo (default: 500).

    Ritorna:
    DecisionTreeClassifier: Il modello di albero di decisione addestrato.
    """
    tree = DecisionTreeClassifier(criterion=criterion,min_samples_split=min_samples_split)
    tree.fit(X, y)
    return tree

def showTree(tree):
    """
    Visualizza graficamente l'albero di decisione e stampa il numero di nodi e foglie.

    Parametri:
    tree (DecisionTreeClassifier): Il modello di albero di decisione addestrato.
    """
    plt.figure(figsize=(40, 30))
    plot_tree(tree, filled=True, fontsize=8, proportion=True)
    plt.show()
    nNodes = tree.tree_.node_count
    nLeaves = tree.tree_.n_leaves
    print("\nThe tree has ", nNodes, "nodes and ", nLeaves, " leaves.\n")

def decisionTreeF1(XTest, YTest, T):
    """
    Calcola l'F1-score di un albero di decisione su un set di test.

    Parametri:
    XTest (DataFrame o array-like): Il set di feature di test.
    YTest (Series o array-like): I valori reali della variabile target.
    T (DecisionTreeClassifier): Il modello di albero di decisione addestrato.

    Ritorna:
    float: L'F1-score calcolato con media pesata.
    """
    yPred = T.predict(XTest)
    f1score = f1_score(YTest, yPred, average='weighted')
    return f1score

def stratifiedKfold(X, y, folds, seed):
    """
    Suddivide il dataset in k-folds stratificati per la validazione incrociata.

    Parametri:
    X (DataFrame o array-like): Le feature indipendenti.
    y (Series o array-like): La variabile target.
    folds (int): Numero di suddivisioni (k).
    seed (int): Seed per la riproducibilità.

    Ritorna:
    tuple: Quattro liste contenenti i dati di addestramento e test per ogni fold.
        - xTrainList (list): Lista dei DataFrame di training.
        - xTestList (list): Lista dei DataFrame di test.
        - yTrainList (list): Lista delle variabili target di training.
        - yTestList (list): Lista delle variabili target di test.
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
    Determina la miglior configurazione per un albero decisionale attraverso validazione incrociata.

    Testa diverse combinazioni del numero di feature e del criterio di impurità ('gini' o 'entropy'),
    valutando le prestazioni con l'F1-score medio.

    Parametri:
    - xTrainList (list): Lista dei set di training suddivisi per ciascun fold.
    - xTestList (list): Lista dei set di test suddivisi per ciascun fold.
    - yTrainList (list): Lista delle etichette di training per ciascun fold.
    - yTestList (list): Lista delle etichette di test per ciascun fold.
    - feature_ranking (list): Lista delle feature ordinate per importanza.

    Ritorna:
    - bestCriterion (str): Il criterio migliore tra 'entropy' e 'gini'.
    - bestNumFeatures (int): Il numero ottimale di feature selezionate.
    - bestF1score (float): Il miglior F1-score ottenuto.
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
    """
    Calcola e visualizza la matrice di confusione per un modello di classificazione,
    stampando anche F1-score e accuratezza media per classe.

    Parametri:
    yTest (array-like): Valori reali delle classi nel set di test.
    yPred (array-like): Predizioni fatte dal modello.
    modelName (str): Nome del modello da usare nel titolo del grafico.
    """
    cm = confusion_matrix(yTest, yPred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix for ' + modelName)
    plt.show()

    target_names = sorted(set(yTest))
    f1_weighted = f1_score(yTest, yPred, average='weighted')

    accuracies = []
    for label in target_names:
        mask = (yTest == label)
        acc = accuracy_score(yTest[mask], yPred[mask])
        accuracies.append(acc)

    average_accuracy = sum(accuracies) / len(accuracies)
    print(f"F1-score pesata: {f1_weighted:.4f}")
    print(f"Accuratezza media: {average_accuracy:.4f}")


def stratified_random_sampling(X, y, n_splits, train_size, num_features):
    """
    Esegue il campionamento casuale stratificato dei dati, mantenendo la distribuzione delle classi
    e selezionando casualmente un sottoinsieme di feature.

    Parametri:
    X (DataFrame): Il dataset delle feature indipendenti.
    y (Series): La variabile target.
    n_splits (int): Numero di suddivisioni da generare.
    train_size (float): Percentuale di dati da includere nel training set (0 < train_size < 1).
    num_features (int): Numero di feature da selezionare casualmente in ogni split.

    Ritorna:
    samples (list): Lista di tuple contenenti i sottoinsiemi (X_train_randomized, y_train).
    selected_features (list): Lista delle feature selezionate per ogni split.
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
    Addestra modelli SVM utilizzando GridSearchCV per ottimizzare i parametri.
    Per ogni campione nei dati, esegue una suddivisione train-validation e cerca
    i migliori iperparametri utilizzando la validazione incrociata.

    Parametri:
    samples (list): Lista di tuple (X_sample, y_sample), dove
        - X_sample è il dataset di input con feature selezionate.
        - y_sample è il target corrispondente.

    Ritorna:
    list: Lista dei migliori modelli SVM trovati per ciascun campione.
    """
    weighted_fscore = make_scorer(f1_score, average='weighted')
    best_models = []

    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
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

        print(f"Modello {i + 1}. Migliori parametri: {grid_search.best_params_}")
        print(f"F-score pesato sul validazione set: {grid_search.best_score_}")
        print("-" * 40)

    return best_models

def majority_voting(models, X, selected_features):
    """
    Esegue il voto di maggioranza tra diversi modelli SVM addestrati.
    Ogni modello fornisce una predizione basata su un sottoinsieme di feature,
    e la classe finale viene determinata scegliendo quella più frequente.

    Parametri:
    models (list): Lista di modelli addestrati.
    X (DataFrame): Dataset di test su cui effettuare le predizioni.
    selected_features (list): Lista delle feature utilizzate da ciascun modello.

    Ritorna:
    np.array: Predizioni finali ottenute tramite voto di maggioranza.
    """
    predictions = []

    for model, features in zip(models, selected_features):
        X_selected = X[features]
        predictions.append(model.predict(X_selected))

    predictions = np.array(predictions).T
    return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predictions)