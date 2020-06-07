import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn import utils
from sklearn.model_selection import train_test_split
import nltk
import multiprocessing
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from WholeOutfits_Evaluator import *

##### Outfit2Vec Doc2Vec
class WholeOutfitsPredictor(object):

    # Tokenisation
    def tokenize_text(self,text):
        tokens = []
        for sent in nltk.sent_tokenize(str(text), language='english'):
            for word in nltk.word_tokenize(sent, language='english'):
                if len(word) < 2:
                    continue
                tokens.append(word)
        return tokens

    def _compute_validation_metrics(self, testing_dataset,  metrics):
        ev = Evaluator(testing_dataset, k=self.k, isStructured =self.isStructured)

        # Complete outfit row not partial
        counter = 0
        for _ in testing_dataset.index:
            if(counter < len(testing_dataset) - 2):
                currentRow = testing_dataset.iloc[counter][0]
                goal = testing_dataset.iloc[counter + 1][0]
                counter += 1
                top_k = self.top_k_recommendations(currentRow,self.methodNumber,self.k)
                ev.add_instance(goal, top_k)

        metrics['hit_ratio'].append(ev.average_hitRatio())
        metrics['precision'].append(ev.average_precision())
        metrics['recall'].append(ev.average_recall())
        metrics['ndcg'].append(ev.average_ndcg())

        return metrics

    # Unused parameters to be removed
    def top_k_recommendations(self, sequence, user_id=None, positiveTags=None, negativeTags=None, exclude=None):
        # Recieves a sequence of (id, outfit, style), and produces k recommendations (as a list of outfit items)
        model = Doc2Vec.load("MODELNAME.model")
        inferred = model.infer_vector(sequence)
        if self.methodNumber == 1:
            # Similar by vector method - based on the words
            top = model.similar_by_vector(inferred,topn=self.k)
        elif self.methodNumber == 2:
            # Most similarity
            top = model.most_similar(positive=[model.infer_vector(sequence)],topn=self.k)
        elif self.methodNumber == 3:
            # similarity based on the doc
            top = model.docvecs.most_similar(positive=[model.infer_vector(sequence)],topn=self.k)

        return top[:self.k]

    def __init__(self, k, window, learning_rate, dataFile, testSize, dm, vector_size, range, resultsFilePath, isStructured, methodNumber,epochs):
        super(WholeOutfitsPredictor, self).__init__()
        self.k = k
        self.window = window
        self.learning_rate= learning_rate
        self.dataFile = dataFile
        self.name = 'Outfits Prediction'
        self.testSize = testSize
        self.dm = dm
        self.vector_size = vector_size
        self.range = range
        self.resultsFilePath = resultsFilePath
        self.isStructured = isStructured
        self.methodNumber = methodNumber
        self.epochs = epochs

        self.metrics = {
            'hit_ratio' : 0,
            'precision': 0,
            'recall': 0,
            'ndcg': 0
        }
        # Read the outfits and style file - pandas
        # col1 - Style , col2 - user name, col3 - outfit
        df_outfits = pd.read_csv(self.dataFile, names={'col1', 'col2', 'col3'})

        self.dataset = df_outfits
        train, test = train_test_split(df_outfits, test_size=self.testSize, random_state=42)
        self.train_tagged = train.apply(lambda r: TaggedDocument(words=self.tokenize_text(r['col3']), tags=[r.col3]), axis=1)
        self.test_tagged = test.apply(lambda r: TaggedDocument(words=self.tokenize_text( r['col3']), tags=[r.col3]), axis=1)

        corpus = []
        for line in self.dataset['col3']:
            words = [x for x in line.split(' ')]
            corpus.append(words)
        self.corpus = corpus

    def train(self):
        cores = multiprocessing.cpu_count()
        model_dbow = Doc2Vec(dm=self.dm, vector_size=self.vector_size, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
        model_dbow.build_vocab([x for x in tqdm(self.train_tagged.values)])

        metrics = {name:[] for name in self.metrics.keys()}

        for i in range(self.range):
            model_dbow.train(utils.shuffle([x for x in tqdm(self.train_tagged.values)]), total_examples=len(self.train_tagged.values), epochs=self.epochs)
            model_dbow.save("NAME.model")
            if(i%3 == 0):
               metrics = self._compute_validation_metrics(self.test_tagged, metrics)
               with open(self.resultsFilePath,'a') as file_writeResults:
                   file_writeResults.write('////// Result for Epoch //////////')
                   file_writeResults.write('\n')
                   file_writeResults.write(str(i))
                   file_writeResults.write('\n')
                   file_writeResults.write('Validation Metrics Values are:')
                   file_writeResults.write('\n')
                   file_writeResults.write(str(metrics['hit_ratio']))
                   file_writeResults.write('\n')
                   file_writeResults.write(str(metrics['precision']))
                   file_writeResults.write('\n')
                   file_writeResults.write(str(metrics['recall']))
                   file_writeResults.write('\n')
                   file_writeResults.write(str(metrics['ndcg']))
                   file_writeResults.write('\n')
                   file_writeResults.write('//////////////////////////////////')
                   file_writeResults.write('\n')
               metrics = {name:[] for name in self.metrics.keys()}

        metrics = self._compute_validation_metrics(self.test_tagged, metrics)
        print("Validation Metrics Values are:")
        print(metrics)
