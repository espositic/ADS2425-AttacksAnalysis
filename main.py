import functions as f
import numpy as np

#Carico in memoria il dataset
trainpath = "trainDdosLabelNumeric.csv"
data = f.load(trainpath)

#Controllo missing values
for col in data.columns:
    missing_count = data[col].isnull().sum()
    print(f"Colonna '{col}' n. {missing_count} missing values")

#Stampo informazioni dataset
shape=data.shape
print(shape)
print(data.head())
print(data.columns)
cols = list(data.columns.values)
f.preElaborationData(data,cols)

#Rimozione colonne inutili (min=max)
data, removedColumns = f.removeColumns(data, cols)
print(removedColumns)

#f.preElaboration(data,'Label')

#Stampo distribuzione classi
f.preElaborationClass(data, 'Label')

#Preparo il dataset per il training
cols = list(data.columns.values)
independentList = cols[0:data.shape[1] - 1]
target = 'Label'
seed=42
np.random.seed(seed)
X = data.loc[:, independentList]
y = data[target]
folds = 5
ListXTrain, ListXTest, ListYTrain, ListYTest = f.stratifiedKfold(X, y, folds, seed)

#Carico in memoria il test set
testpath = "testDdosLabelNumeric.csv"
testset = f.load(testpath)
yTest = testset['Label']


# ***************************************************
# ******* DECISION TREE CON MUTUAL INFO RANK ********
# ***************************************************

#Calcolo Mutual Info Rank
rankMI = f.mutualInfoRank(data, independentList, target)

#Selezione le prime 10 feature e la classe
N = 10
toplist = f.topFeatureSelect(rankMI, N)
toplist.append(target)

print("Prime 10 Feature Mutual Info Rank")
print(toplist)
selectedMIData=data.loc[:, toplist]
print(selectedMIData.shape)
print(selectedMIData.head())
print(selectedMIData.columns)

#Determino migliore configurazione per l'albero decisionale
bestCriterion, bestNumFeatures, bestF1score = f.determineDecisionTreeFoldConfiguration(
    ListXTrain, ListXTest, ListYTrain, ListYTest, toplist[:-1]
)
print("\nMigliore configurazione trovata:")
print("Criterio:", bestCriterion)
print("Numero di feature selezionate:", bestNumFeatures)
print("Miglior F1 Score:", bestF1score)

#Addestro l'albero decisionale con la migliore configurazione sull'intero dataset
bestTree = f.decisionTreeLearner(X[toplist[:bestNumFeatures]], y, bestCriterion)
f.showTree(bestTree)

# Seleziono le feature usate per addestrare l'albero e le applico al test set per la valutazione
XTest = testset[toplist[:bestNumFeatures]]
yPred = bestTree.predict(XTest)
modelName = "Decision Tree with Mutual Info Rank"
f.computeConfusionMatrix(yTest, yPred, modelName)


# **************************************************
# ******* DECISION TREE CON INFORMATION GAIN *******
# **************************************************

#Calcolo Information Gain
rankGI = f.giRank(data, independentList, target)

#Selezione le prime 10 feature e la classe
N=10
toplist = f.topFeatureSelect(rankGI, N)
toplist.append(target)

print("Prime 10 Feature Information Gain")
print(toplist)
selectedGIData=data.loc[:, toplist]
print(selectedGIData.shape)
print(selectedGIData.head())
print(selectedGIData.columns)
bestCriterion, bestNumFeatures, bestF1score=f.determineDecisionTreeFoldConfiguration(
    ListXTrain,ListXTest,ListYTrain,ListYTest, toplist[:-1]
)
print("\nMigliore configurazione trovata:")
print("Criterio:", bestCriterion)
print("Numero di feature selezionate:", bestNumFeatures)
print("Miglior F1 Score:", bestF1score)

# Addestro l'albero decisionale con la migliore configurazione sull'intero dataset
bestTree = f.decisionTreeLearner(X[toplist[:bestNumFeatures]], y, bestCriterion)
f.showTree(bestTree)

# Seleziono le feature usate per addestrare l'albero e le applico al test set per la valutazione
XTest=testset[toplist[:bestNumFeatures]]
yPred = bestTree.predict(XTest)
modelName = "Decision Tree with Information Gain"
f.computeConfusionMatrix(yTest, yPred, modelName)


# *************************************************
# ************* DECISION TREE CON PCA *************
# *************************************************

# Applico PCA al dataset
X=data.loc[:, independentList]
pca,pcalist=f.pca(X)
pcaData=f.applyPCA(X,pca,pcalist)
pcaData.insert(loc=len(independentList), column=target, value=data[target], allow_duplicates =True)
print(pcaData.columns.values)
print(pcaData.head())
print(pcaData.shape)

# Seleziono le prime 10 componenti principali
N = 10
selectedPCAData = pcaData.iloc[:, :N]
print(selectedPCAData)
ListXTrain,ListXTest,ListYTrain,ListYTest=f.stratifiedKfold(selectedPCAData,y,folds, seed)

#Determino migliore configurazione per l'albero decisionale
pca_toplist=selectedPCAData.columns
bestCriterion, bestNumFeatures, bestF1score=f.determineDecisionTreeFoldConfiguration(
    ListXTrain,ListXTest,ListYTrain,ListYTest, pca_toplist
)
print("\nMigliore configurazione trovata:")
print("Criterio:", bestCriterion)
print("Numero di feature selezionate:", bestNumFeatures)
print("Miglior F1 Score:", bestF1score)

# Addestro l'albero decisionale con la migliore configurazione sull'intero dataset
bestTree = f.decisionTreeLearner(selectedPCAData[pca_toplist[:bestNumFeatures]], y, bestCriterion)
f.showTree(bestTree)

# Seleziono le feature usate per addestrare l'albero e applico la PCA al test set per la valutazione
XTest_PCA = f.applyPCA(testset.loc[:, independentList],pca,pcalist)
yPred = bestTree.predict(XTest_PCA[pca_toplist[:bestNumFeatures]])
modelName = "Decision Tree with PCA"
f.computeConfusionMatrix(yTest, yPred, modelName)


# *************************************************
# **************** ENSEMBLE 10 SVM ****************
# *************************************************

# Campionamento stratificato per ottenere 10 sample,
# ciascuno con il 80% dei dati originali e 20 feature selezionate casualmente
samples, selected_features = f.stratified_random_sampling(X, y, n_splits=10, train_size=0.8, num_features=20)

# Addestramento di 10 modelli SVM sui sottoinsiemi generati
models = f.train_svm(samples)
YPredEnsemble = f.majority_voting(models, testset, selected_features)
modelName = "Ensemble 10 SVM"
# Computazione e visualizzazione della matrice di confusione
f.computeConfusionMatrix(yTest, YPredEnsemble, modelName)