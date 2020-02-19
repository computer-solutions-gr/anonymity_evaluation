import json
import pandas
import numpy


def load_and_prepare(file, set_index=False):

    if set_index:
        data = pandas.read_csv(file, index_col=[0])
    else:
        data = pandas.read_csv(file)
    print(data.head(2))

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


def load_metrics(k, qi):
    filename = 'data/qi={0}/metrics_k={1}.txt'.format(qi, k)

    with open(filename) as json_file:
        data = json.load(json_file)

    return {'GIL': data['GIL'], 'DM': data['DM'], 'C_AVG': data['C_AVG']}
