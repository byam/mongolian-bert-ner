from flair import TaggedCorpus
from flair import NLPTaskDataFetcher

# define columns
columns = {0: 'text', 3: 'ner'}

# this is the folder in which train, test and dev files reside
data_folder = '../data/'

# retrieve corpus using column format, data folder and the names of the train, dev and test files
corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
                                                             train_file='train.txt',
                                                             test_file='test.txt',
                                                             dev_file='valid.txt')
