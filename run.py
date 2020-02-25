import numpy
import pandas
import datetime
from lib.ml import X_Y, separate_train_test, one_hot_encode, set_last_column
from lib.cross_validation import Cross_Validation
from lib.helpers import load_and_prepare, load_metrics
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression
from run_parameters import K_S, QI_S, ALGOS

random_state = 7
numpy.random.seed(random_state)


def run(model, features, labels, k, qi):

    train_scores, validation_scores, test_scores_auc, test_scores_mcc = Cross_Validation(
        model, 10, features, labels)
    print(k, model.__str__().split("(")[0])
    print('Train AUC', float(format(numpy.mean(train_scores), '.3f')))
    print('Validation AUC', float(format(numpy.mean(validation_scores), '.3f')))
    print('Test AUC', float(format(numpy.mean(test_scores_auc), '.3f')))
    print('Test MCC', float(format(numpy.mean(test_scores_mcc), '.3f')))
    if qi == 0:
        metrics = {'GIL': 0, 'DM': 0, 'C_AVG': 0}
    else:
        metrics = load_metrics(k, qi)
    # print()
    return {
        'k': k,
        'qi': qi,
        'Start Time': datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
        'Classifier': model.__str__().split("(")[0],
        'Classifier(Full)': model.__str__(),
        'Random State': random_state,
        'Train AUC': float(format(numpy.mean(train_scores), '.3f')),
        'Validation AUC': float(format(numpy.mean(validation_scores), '.3f')),
        'Test AUC': float(format(numpy.mean(test_scores_auc), '.3f')),
        'Test MCC': float(format(numpy.mean(test_scores_mcc), '.3f')),
        'GIL': metrics['GIL'],
        'DM': metrics['DM'],
        'C_AVG': metrics['C_AVG']
    }


# Create Files Array
k_values = K_S
qi_values = QI_S
files = []
files.append({'k': 1, 'qi': 0, 'file': 'data/data.csv'})
for qi in qi_values:
    for k in k_values:
        filename = 'data/qi={0}/data_k={1}.csv'.format(qi, k)
        files.append({'k': k, 'qi': qi, 'file': filename})
print(files)
# files = [
#     {'k': 1, 'qi': 0, 'file': 'data/data.csv'},
#     {'k': 3, 'qi': 3, 'file': 'data/qi=3/data_k=3.csv'},
#     {'k': 5, 'qi': 3, 'file': 'data/qi=3/data_k=5.csv'},
#     {'k': 10, 'qi': 3, 'file': 'data/qi=3/data_k=10.csv'},
#     {'k': 15, 'qi': 3, 'file': 'data/qi=3/data_k=15.csv'},
#     {'k': 20, 'qi': 3, 'file': 'data/qi=3/data_k=20.csv'},
#     {'k': 3, 'qi': 2, 'file': 'data/qi=2/data_k=3.csv'},
#     {'k': 5, 'qi': 2, 'file': 'data/qi=2/data_k=5.csv'},
#     {'k': 10, 'qi': 2, 'file': 'data/qi=2/data_k=10.csv'},
#     {'k': 15, 'qi': 2, 'file': 'data/qi=2/data_k=15.csv'},
#     {'k': 20, 'qi': 2, 'file': 'data/qi=2/data_k=20.csv'},
# ]

results = pandas.DataFrame(columns=[
    'k',
    'qi',
    'Start Time',
    'Classifier',
    'Classifier(Full)',
    'Random State',
    'Train AUC',
    'Validation AUC',
    'Test AUC',
    'Test MCC',
    'GIL',
    'DM',
    'C_AVG'
])
for file in files:
    print(file)
    if file['k'] != 1:
        data = load_and_prepare(file['file'])
        encoded_fields = ['AGE', 'SEX', 'OUTCOME']
        if file['qi'] > 3:
            encoded_fields.append('CURADM_DAYS')
        if file['qi'] > 4:
            encoded_fields.append('PREVADM_DAYS')
        data = one_hot_encode(data, encoded_fields)
        data = set_last_column(data, 'READMISSION_30_DAYS')
    else:
        data = load_and_prepare(file['file'], True)
    X, Y = X_Y(data)
    X_train, X_test, Y_train, Y_test = separate_train_test(X, Y)
    models = ALGOS

    for model in models:
        results = results.append(
            run(model, X_test, Y_test, file['k'], file['qi']), ignore_index=True)

print(results)
results.to_pickle('results/latest.pkl')
with pandas.ExcelWriter('results/experiments.xlsx', mode='a') as writer:
    results.to_excel(
        writer, sheet_name=datetime.datetime.now().strftime('%Y%m%d%H%M%S'), float_format="%.3f")
