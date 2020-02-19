import numpy
import pandas
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection


def X_Y(data):
    ''' Split features and labels
    TODO: More doc
    '''
    dataset = data.values
    numpy.random.shuffle(dataset)
    column_count = dataset.shape[1]
    print(column_count)
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:(column_count - 1)].astype(float)
    Y = dataset[:, (column_count - 1)]
    return X, Y


def separate_train_test(X, Y, test_size=0.1, scale=True, random_state=7):
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X, Y, test_size=test_size, random_state=random_state)
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, Y_train, Y_test


def one_hot_encode(data, columns=[]):
    for column in columns:
        data = data.join(pandas.get_dummies(data[column], prefix=column))
        data.drop(column, axis=1, inplace=True)
    return data


def set_last_column(data, column_name):
    cols = [col for col in data if col != column_name] + [column_name]
    data = data[cols]
    print(data.head(2))
    return data
