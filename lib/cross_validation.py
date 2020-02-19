import numpy
from math import floor
from sklearn.metrics import roc_auc_score, matthews_corrcoef


def partition(vector, fold, k):
    size = vector.shape[0]
    start = floor((size / k) * fold)
    end = floor((size / k) * (fold + 1))
    validation = vector[start:end]
    # print(str(type(vector)))

    if str(type(vector)) == "<class 'scipy.sparse.csr.csr_matrix'>":
        indices = range(start, end)
        mask = numpy.ones(vector.shape[0], dtype=bool)
        mask[indices] = False
        training = vector[mask]
    elif str(type(vector)) == "<class 'numpy.ndarray'>":
        training = numpy.concatenate((vector[:start], vector[end:]))
    return training, validation


def Cross_Validation(learner, k, examples, labels):
    train_folds_score = []
    validation_folds_score = []
    test_score_auc = []
    test_score_mcc = []
    for fold in range(0, k):
        training_set, validation_set = partition(examples, fold, k)
        training_labels, validation_labels = partition(labels, fold, k)
        learner.fit(training_set, training_labels)
        training_predicted = learner.predict(training_set)
        validation_predicted = learner.predict(validation_set)
        test_predicted = learner.predict(examples)

        train_folds_score.append(roc_auc_score(
            training_labels, training_predicted))
        validation_folds_score.append(roc_auc_score(
            validation_labels, validation_predicted))
        test_score_auc.append(roc_auc_score(labels, test_predicted))
        test_score_mcc.append(matthews_corrcoef(labels, test_predicted))

    return train_folds_score, validation_folds_score, test_score_auc, test_score_mcc
