from flair.data import TaggedCorpus
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, BytePairEmbeddings
from flair.models import SequenceTagger
from typing import List

import gensim
# from smart_open import open
#
#
# with open('https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.mn.300.vec.gz') as fin:
#     with open('mn_embeddings.txt', 'w') as fout:
#         for line in fin:
#             fout.write(line)


# word_vectors = gensim.models.KeyedVectors.load_word2vec_format('mn_embeddings.txt', binary=False)
# word_vectors.save('converted')
# MNembedding = WordEmbeddings('converted')

# define columns
columns = {0: 'text', 3: 'ner'}

# this is the folder in which train, test and dev files reside
data_folder = '../data/'

# retrieve corpus using column format, data folder and the names of the train, dev and test files
corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
                                                             train_file='train.txt',
                                                             test_file='test.txt',
                                                             dev_file='valid.txt',)

print(len(corpus.train))
print(corpus.train[-1].to_tagged_string('text'))
print(corpus.train[-1].to_tagged_string('ner'))

tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)
# tag_dictionary.idx2item = ['<unk>', 'O', 'B-PER', 'B-MISC', 'I-MISC', 'I-PER', 'B-ORG', 'B-LOC', 'I-LOC', 'I-ORG',
#                            '<START>', '<STOP>']
# print(tag_dictionary.idx2item)

# embedding_types: List[TokenEmbeddings] = [
#
#     # BytePairEmbeddings('mn'),
#     # WordEmbeddings('en'),
#
#     # comment in this line to use character embeddings
#     CharacterEmbeddings(),
#
#     # comment in these lines to use flair embeddings
#     # FlairEmbeddings('news-forward'),
#     # FlairEmbeddings('news-backward'),
# ]
# embeddings = WordEmbeddings('converted')
# embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
embeddings = BytePairEmbeddings('mn')

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type)

# 6. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/example-ner',
              learning_rate=0.001,
              mini_batch_size=32,
              max_epochs=150)

# 8. plot training curves (optional)
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves('resources/taggers/example-ner/loss.tsv')
plotter.plot_weights('resources/taggers/example-ner/weights.txt')