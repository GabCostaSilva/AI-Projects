from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


class Classifier:
    def __init__(self, data, target, data_length):
        self.data = data
        self.target = target

        self.predicted_classes = dict()
        self.predicted_classes["tree"] = np.zeros(data_length)
        self.predicted_classes["knn"] = np.zeros(data_length)
        self.predicted_classes["bayes"] = np.zeros(data_length)
        self.predicted_classes["regression"] = np.zeros(data_length)
        self.predicted_classes["mlp"] = np.zeros(data_length)

        self.stats = dict()
        self.stats["tree"] = {"f1_score": list(), "accuracy_score": list()}
        self.stats["knn"] = {"f1_score": list(), "accuracy_score": list()}
        self.stats["bayes"] = {"f1_score": list(), "accuracy_score": list()}
        self.stats["regression"] = {"f1_score": list(), "accuracy_score": list()}
        self.stats["mlp"] = {"f1_score": list(), "accuracy_score": list()}

    def run(self, decision_tree_params, knn_params, nb_params, mlp_params, regression_params):
        # treinamento
        inner_skf = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
        # ajuste dos params
        outer_skf = model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=None)

        decision_tree = DecisionTreeClassifier(**self.decision_tree_tune_params(outer_skf, decision_tree_params))
        knn = KNeighborsClassifier(**self.knn_tune_params(outer_skf, knn_params))
        bayes = GaussianNB(**self.nb_tune_params(outer_skf, nb_params))
        regression = LogisticRegression(**self.logistic_regression_tune_params(outer_skf, regression_params))
        mlp = MLPClassifier(**self.mlp_tune_params(outer_skf, mlp_params))

        for train, test in inner_skf.split(self.data, self.target):
            data_train, target_train = self.data[train], self.target[train]
            data_test, target_test = self.data[test], self.target[test]

            decision_tree = decision_tree.fit(data_train, target_train)
            decision_tree_predicted = decision_tree.predict(data_test)
            self.predicted_classes["tree"][test] = decision_tree_predicted
            self.stats["tree"]["f1_score"].append(f1_score(self.target[test], decision_tree_predicted, average=None))
            self.stats["tree"]["accuracy_score"].append(accuracy_score(self.target[test], decision_tree_predicted, average=None))

            knn = knn.fit(data_train, target_train)
            knn_predicted = knn.predict(data_test)
            self.predicted_classes["knn"][test] = knn_predicted
            self.stats["knn"]["f1_score"].append(f1_score(self.target[test], knn_predicted, average=None))
            self.stats["knn"]["accuracy_score"].append(accuracy_score(self.target[test], knn_predicted, average=None))

            bayes = bayes.fit(data_train, target_train)
            bayes_predicted = bayes.predict(data_test)
            self.predicted_classes["bayes"][test] = bayes_predicted
            self.stats["bayes"]["f1_score"].append(f1_score(self.target[test], bayes_predicted, average=None))
            self.stats["bayes"]["accuracy_score"].append(accuracy_score(self.target[test], bayes_predicted, average=None))

            regression = regression.fit(data_train, target_train)
            regression_predicted = regression.predict(data_test)
            self.predicted_classes["regression"][test] = regression_predicted
            self.stats["regression"]["f1_score"].append(f1_score(self.target[test], regression_predicted, average=None))
            self.stats["regression"]["accuracy_score"].append(accuracy_score(self.target[test], regression_predicted, average=None))

            mlp = mlp.fit(data_train, target_train)
            mlp_predicted = mlp.predict(data_test)
            self.predicted_classes["mlp"][test] = mlp_predicted
            self.stats["mlp"]["f1_score"].append(f1_score(self.target[test], mlp_predicted, average=None))
            self.stats["mlp"]["accuracy_score"].append(accuracy_score(self.target[test], mlp_predicted, average=None))

    def graph_plot(self):
        f1_means = []
        f1_std = []
        accuracy_means = []
        accuracy_std = []

        for classifier in self.stats.keys():
            f1_means.append(np.nanmean(np.asarray(self.stats[classifier]["f1_score"])))
            f1_std.append(np.std(np.asarray(self.stats[classifier]["f1_score"])))
            accuracy_means.append(np.nanmean(np.asarray(self.stats[classifier]["accuracy_score"])))
            accuracy_std.append(np.std(np.asarray(self.stats[classifier]["accuracy_score"])))

        labels, y = zip(*self.stats.items())

        x = np.arange(len(list(labels)))
        width = 0.35
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, f1_means, width, label='F1 Mean')
        rects2 = ax.bar(x + width / 2, accuracy_means, width, label='Accuracy Mean')

        ax.set_ylabel('Scores')
        ax.set_title('Medias por algoritmo')
        ax.set_title('Medias')
        ax.set_xticks(x)
        ax.set_xticklabels(list(labels))
        ax.legend()

        self.autolabel(rects1, ax)
        self.autolabel(rects2, ax)
        fig.tight_layout()
        fig.set_size_inches(13, 6)

        fig, ax = plt.subplots()

        rects1 = ax.bar(x - width / 2, f1_std, width, label='F1 Variance')
        rects2 = ax.bar(x + width / 2, accuracy_std, width, label='Accuracy Variance')

        ax.set_ylabel('Scores')
        ax.set_title('Desvio Padrao por algoritmo')
        ax.set_title('Desvio Padrao')
        ax.set_xticks(x)
        ax.set_xticklabels(list(labels))
        ax.legend(loc='best')

        self.autolabel(rects1, ax)
        self.autolabel(rects2, ax)
        fig.tight_layout()
        fig.set_size_inches(13, 6)
        plt.show()

    def autolabel(self, rects, ax):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(round(height, 2)),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(3, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    def show_report(self):
        for classifier in self.predicted_classes.keys():
            print(f"Resultados do classificador {classifier}:\n")
            print(metrics.classification_report(self.target, self.predicted_classes[classifier]))
            print()
            print("Matriz de confusao\n")
            print(metrics.confusion_matrix(self.target, self.predicted_classes[classifier]))

    def knn_tune_params(self, outer_skf, params):
        knn = KNeighborsClassifier()

        grid_search = model_selection.GridSearchCV(
            knn, param_grid=params, cv=outer_skf, scoring='accuracy', refit=False)

        grid_search.fit(self.data, self.target)

        knn_best_params = grid_search.best_params_

        print(f"KNN:\n{knn_best_params}")

        return grid_search.best_params_

    def decision_tree_tune_params(self, outer_skf, params):
        decision_tree = DecisionTreeClassifier()

        grid_search = model_selection.GridSearchCV(
            decision_tree, param_grid=params, cv=outer_skf, scoring='accuracy', refit=False
        )

        grid_search.fit(self.data, self.target)

        print(f"Árvore de Decisão:\n{grid_search.best_params_}")

        return grid_search.best_params_

    def logistic_regression_tune_params(self, outer_skf, params):
        regression = LogisticRegression()

        grid_search = model_selection.GridSearchCV(
            regression, param_grid=params, cv=outer_skf, scoring='accuracy', refit=False
        )

        grid_search.fit(self.data, self.target)

        print(f"Regressão Logística:\n{grid_search.best_params_}")

        return grid_search.best_params_

    def mlp_tune_params(self, outer_skf, params):
        mlp = MLPClassifier()

        grid_search = model_selection.GridSearchCV(
            mlp, param_grid=params, cv=outer_skf, scoring='accuracy', refit=False
        )

        grid_search.fit(self.data, self.target)

        print(f"Multi-layer Perceptron:\n{grid_search.best_params_}")

        return grid_search.best_params_

    def nb_tune_params(self, outer_skf, params):
        bayes = GaussianNB()

        grid_search = model_selection.GridSearchCV(
            bayes, param_grid=params, cv=outer_skf, scoring='accuracy', refit=False
        )

        grid_search.fit(self.data, self.target)

        print(f"Naive Bayes: \n{grid_search.best_params_}")

        return grid_search.best_params_
