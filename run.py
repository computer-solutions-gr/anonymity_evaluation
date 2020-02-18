import numpy
import pandas
import datetime
from math import floor
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

random_state = 7
numpy.random.seed(random_state)


def X_Y(data):
    dataset = data.values
    numpy.random.shuffle(dataset)
    column_count = dataset.shape[1]
    print(column_count)
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:(column_count - 1)].astype(float)
    Y = dataset[:, (column_count - 1)]
    return X, Y


def separate_train_test(X, Y, test_size=0.1):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X, Y, test_size=test_size, random_state=7)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test


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
        test_predicted = learner.predict(X_test)

        train_folds_score.append(roc_auc_score(
            training_labels, training_predicted))
        validation_folds_score.append(roc_auc_score(
            validation_labels, validation_predicted))
        test_score_auc.append(roc_auc_score(Y_test, test_predicted))
        test_score_mcc.append(matthews_corrcoef(Y_test, test_predicted))

    return train_folds_score, validation_folds_score, test_score_auc, test_score_mcc


def load_and_prepare(file, set_index=False):

    if set_index:
        data = pandas.read_csv(file, index_col=[0])
    else:
        data = pandas.read_csv(file)
    print(data.head())

    zero_indices = data[data['READMISSION_30_DAYS'] == 0].index

    sample_size_to_remove = sum(
        data['READMISSION_30_DAYS'] == 0) - sum(data['READMISSION_30_DAYS'] == 1)
    random_indices = numpy.random.choice(
        zero_indices, sample_size_to_remove, replace=False)
    data = data.drop(random_indices)
    print(len(data))
    readmission_count = data.groupby(
        'READMISSION_30_DAYS').size().sort_values(ascending=False)
    print(readmission_count)

    return data


def one_hot_encode(data):
    data = data.join(pandas.get_dummies(data['AGE'], prefix='AGE'))
    data.drop('AGE', axis=1, inplace=True)

    data = data.join(pandas.get_dummies(data['OUTCOME'], prefix='OUTCOME'))
    data.drop('OUTCOME', axis=1, inplace=True)

    data = data.join(pandas.get_dummies(data['SEX'], prefix='SEX'))
    data.drop('SEX', axis=1, inplace=True)

    cols = [col for col in data if col !=
            'READMISSION_30_DAYS'] + ['READMISSION_30_DAYS']
    data = data[cols]
    print(data.head())
    return data


def run(model, features, labels, k):

    train_scores, validation_scores, test_scores_auc, test_scores_mcc = Cross_Validation(
        model, 10, features, labels)
    # print(train_scores, validation_scores, test_scores)
    print(model)
    print('Train AUC', float(format(numpy.mean(train_scores), '.3f')))
    print('Validation AUC', float(format(numpy.mean(validation_scores), '.3f')))
    print('Test AUC', float(format(numpy.mean(test_scores_auc), '.3f')))
    print('Test MCC', float(format(numpy.mean(test_scores_mcc), '.3f')))
    print()
    return {
        'k': k,
        'Start Time': datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
        'Classifier': model.__str__().split("(")[0],
        'Classifier(Full)': model.__str__(),
        'Random State': random_state,
        'Train AUC': float(format(numpy.mean(train_scores), '.3f')),
        'Validation AUC': float(format(numpy.mean(validation_scores), '.3f')),
        'Test AUC': float(format(numpy.mean(test_scores_auc), '.3f')),
        'Test MCC': float(format(numpy.mean(test_scores_mcc), '.3f'))
    }


files = [
    {'k': 1, 'file': 'data/data.csv'},
    # {'k': 3, 'file': 'data/data_k=3.csv'}
]

for file in files:
    if file['k'] != 1:
        data = load_and_prepare(file['file'])
        data = one_hot_encode(data)
    else:
        data = load_and_prepare(file['file'], True)
    X, Y = X_Y(data)
    X_train, X_test, Y_train, Y_test = separate_train_test(X, Y)
    models = [LogisticRegression(solver='liblinear'), KNeighborsClassifier(
    ), GaussianNB(), SVC(gamma='auto')]  # LogisticRegression(solver='liblinear')
    results = pandas.DataFrame(columns=[
        'k',
        'Start Time',
        'Classifier',
        'Classifier(Full)',
        'Random State',
        'Train AUC',
        'Validation AUC',
        'Test AUC',
        'Test MCC'
    ])
    for model in models:
        results = results.append(run(model, X_test, Y_test, file['k']), ignore_index=True)

    print(results)
    with pandas.ExcelWriter('results/experiments.xlsx', mode='a') as writer:
        results.to_excel(writer, sheet_name=datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
