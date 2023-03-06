;;; Imports
(import matplotlib.pyplot :as plt)
;(import numpy :as np)
;(import pandas :as pd)
(import sklearn [datasets]
        sklearn.ensemble [RandomForestClassifier]
        sklearn.model_selection [train_test_split]
        sklearn.metrics [precision_score confusion_matrix recall_score f1_score precision_recall_fscore_support ConfusionMatrixDisplay]
        sklearn.svm [SVC])

;;; Loas Iris
(setv data_iris (datasets.load_iris))

;;; SKLEARN classifiers
(setv clf (RandomForestClassifier))
(setv clf2 (SVC))

;;; Define split train test
(defn split-train-test [data [test_size 0.25]]
    (let [X (py "data_iris.data[:, :2]")
          Y data.target
         ]
    (return [X Y (train_test_split X Y :test_size test_size)])
    )
)

;;; Define classifier evaluation function
(defn use-sklearn-class [x_train x_test y_train y_test clf] 
    (let [clf_trained (clf.fit x_train y_train)
          y_predict (clf_trained.predict x_test)
          precision (precision_score y_test y_predict :average "weighted")
          recall (recall_score y_test y_predict :average "weighted")
          f1 (f1_score y_test y_predict :average "weighted")
          prfs (precision_recall_fscore_support y_test y_predict :average "weighted")
          conf_mat (ConfusionMatrixDisplay.from_predictions y_test y_predict)
         ]
    (plt.title clf)
    (print f"\n Precision: {precision} \n Recall: {recall} \n F1: {f1} \n PRFS: {prfs}")
    (return y_predict)
    )
)

;; Split
(setv [X Y [x_train x_test y_train y_test]] (split-train-test data_iris))

;; Evaluate
(use-sklearn-class x_train x_test y_train y_test clf)
(use-sklearn-class x_train x_test y_train y_test clf2)

;; Plot Iris data distribution
(plt.figure)
(plt.scatter (py "X[:, 0]") (py "X[:, 1]") :c Y :cmap plt.cm.Set1 :edgecolor "k")
(plt.xlabel "Sepal length")
(plt.ylabel "Sepal width")
(plt.savefig "./iris_classes_scatter.png")
(plt.show)