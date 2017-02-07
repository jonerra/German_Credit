# Imports
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.ensemble import ExtraTreesClassifier


def explore(file, target):
    df = pd.read_csv(file)

    if len(set(df[target])) < 11:
        chi = []
        for column in df:
            obs = np.array([df[column], df[target]])
            chi.append(chi2_contingency(obs)[0])

        labels = []
        y = chi
        for i in range(len(y)):
            labels.append(i)
        plt.bar(labels, y, width=0.7, align='center')
        plt.xticks(labels, df.columns, rotation=90)
        plt.xlabel('Variables')
        plt.ylabel('Chi-Square')
        plt.title('Chi-Square Plot')
        plt.show()
    else:
        correlations = df.corr()
        cax = plt.matshow(correlations, vmin=-1, vmax=1)
        plt.colorbar(cax)
        plt.figure(figsize=(15, 15))
        plt.show()

    # Our variables for the classification task
    X = np.array(df.drop([target], 1))
    y = np.array(df[target])

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    columns = []
    for i in indices:
        name = df.columns[i]
        columns.append(name)
    # print(columns)

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. %s (%f)" % (f + 1, columns[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices], color='r', align="center")
    plt.xticks(range(X.shape[1]), columns, rotation='90')
    plt.xlim([-1, X.shape[1]])
    plt.show()

explore('data/german_num_data_train.csv', 'fraud')
