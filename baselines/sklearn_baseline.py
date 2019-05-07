import sklearn_crfsuite
from seqeval.metrics import classification_report
from sklearn_crfsuite import metrics
from collections import Counter


def get_data(filename):
    data = []
    with open(f'../data/{filename}', 'r') as file:
        tokens = []
        labels = []
        for line in file:
            if not line == '\n':
                tokens.append(line.split()[0])
                labels.append(line.split()[-1])
            else:
                data.append((tokens, labels))
                tokens = []
                labels = []
    return data


train_data = get_data('train.txt')
test_data = get_data('test.txt')
valid_data = get_data('valid.txt')

classes = list(set([item for sublist in [i[1] for i in train_data] for item in sublist]))
classes.remove('O')

print(Counter([item for sublist in [i[1] for i in test_data] for item in sublist]))
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit([i[0] for i in train_data], [i[1] for i in train_data])

# valid
# y_pred = crf.predict([i[0] for i in valid_data])
# print(classification_report([i[1] for i in valid_data], y_pred, labels=classes))
# test
y_pred = crf.predict([i[0] for i in test_data])
assert len([i[1] for i in test_data]) == len(y_pred)
print(classification_report([i[1] for i in test_data], y_pred))
