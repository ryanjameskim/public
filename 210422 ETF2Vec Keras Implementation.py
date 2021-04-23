'''Uses the Word2Vec concept with individual stock holdings from all of BlackRock's
US Public Equity ETF  holdings in order to dimensionalize 'ETF stock selection criteria'
into vector form. Proximity in holding size within an ETF is taken as primary context with a Zipf
similiarity negative sample selection. Uses Keras functional API.

Adapted for ETF differentiation from original Word2Vec from
https://adventuresinmachinelearning.com/word2vec-keras-tutorial/'''

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import skipgrams
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Reshape, Embedding, Dense
from tensorflow.keras import Model

#import data
df = pd.read_pickle(
    r'C:\Users\ryanj\Google Drive\ML\Ishares Data\210331\Holdings_small\Aggregate.pkl')
#df = pd.read_csv('/content/Aggregate.csv')

df = df.sort_values(['ETF', 'Weight (%)'], ascending=[True, False])


#Find tickers and frequencies
val_cnts = df['Ticker'].value_counts() 
etf_names = df['ETF'].value_counts().index.to_numpy()
np.random.seed(42)
np.random.shuffle(etf_names)

#Create vocabulary and integerization of stock tickers
vocab_size = len(val_cnts)  #3866
idx_to_tick = {i + 1: ticker for i, ticker in enumerate(val_cnts.keys())}  #natural counting
tick_to_idx = {ticker : index for index, ticker in idx_to_tick.items()}   #inverse

#constants
window_size = 3    # three nearest in portfolio % weight to the stock in question
vector_dim = 128    # dimensionality of the embedding vector
epochs = 10 ** 6    # one million epochs
negative_samples = 5 

#validation constants
valid_size = 16
valid_window = 100
#valid_examples = np.random.choice(valid_window, valid_size, replace=False)
#['WAT', 'SWKS', 'PAYX', 'CSCO', 'RF', 'NKE', 'INTC', 'ALL', 'GS', 'AVGO', 'HUM', 'NEE', 'T', 'CL', 'DVA', 'AMGN']
custom_examples = ['CSCO', 'NKE', 'INTC', 'GS', 'T', 'TSLA', 'AAPL', 'PAYX']
valid_examples = [tick_to_idx[tick] for tick in custom_examples]
#picks {16} of the first {100} words for validation
#may need to replace this with some pure play companies like TSLA compared to Ford

#sampling table for negative examples
sampling_table = sequence.make_sampling_table(vocab_size + 1)  #+1 due to index zero being skipped

#function to create skipgrams per ETF
targets, contexts, labels = [], [], []
for etf in etf_names:
    tokens = np.array([tick_to_idx[tick] for tick in df.loc[df['ETF'] == etf, 'Ticker'].values]) 
    etf_couples, etf_labels = skipgrams(
                        tokens,
                        vocab_size,
                        window_size = window_size,
                        negative_samples = negative_samples,
                        sampling_table = sampling_table)
    etf_targets, etf_contexts = zip(*etf_couples)   #separate into target and contexts by etf
    targets.append(np.array(etf_targets))
    contexts.append(np.array(etf_contexts))
    labels.append(np.array(etf_labels))
#may need to add criteria for context negative selection to random XX% in weight away from target word

#flatten to a single numpy
targets = np.concatenate(targets).ravel()
contexts = np.concatenate(contexts).ravel()
labels = np.concatenate(labels).ravel()

#**MODEL**
# create some input variables
input_target = tf.keras.Input((1,))
input_context = tf.keras.Input((1,))

embedding = Embedding(vocab_size + 1, vector_dim, input_length=1, name='embedding') #vocab_size + 1, see doc

#create target and context embedding layers
target_embedding = embedding(input_target)
target_embedding = Reshape((vector_dim, 1))(target_embedding)
context_embedding = embedding(input_context)
context_embedding = Reshape((vector_dim, 1))(context_embedding)

#perform dot product
dot_product = tf.keras.layers.dot([target_embedding, context_embedding], axes = 1) 
dot_product = Reshape((1,))(dot_product)

#output layer
output = Dense(1, activation='sigmoid')(dot_product)

#create the primary training model
model = Model(inputs=[input_target, input_context], outputs=output)
model.compile(loss='binary_crossentropy', optimizer='adam')

#cosine similarity function for validation
similarity = tf.keras.layers.dot([target_embedding, context_embedding], normalize=True, axes = 1)

#validation model for using the similarity operation (very slow)
validation_model = Model(inputs=[input_target, input_context], outputs=similarity)

#Similarity Call Back
class SimilarityCallback:
    def run_sim(self):
        for i, valid_ex_id in enumerate(valid_examples):       #goes through each of the valid examples
            valid_name = idx_to_tick[valid_ex_id]       #find name of validation ticker 
            top_k = 8                                # number of nearest neighbors
            sim = self._get_sim(valid_ex_id)        #get similarity against all other tickers
            nearest = (-sim).argsort()[1:top_k + 1]  #gets negative of the cosine scores, lists the indexes of the most negative to positve scores
                                                     #then indexing the argsort lists the smallest numbers, skips index 0 because it will be -1 (the cosine similarity to itself) 
            log_str = f'Nearest to {valid_name}:'
            for kth_near in nearest:
                close_tick = idx_to_tick[kth_near]
                log_str = f'{log_str} {close_tick},'  #keeps appending to the log string
            print(log_str)
    @staticmethod
    def _get_sim(valid_ex_id):
        sim = np.zeros((vocab_size,))  #initalize sim and inputs
        validation_ticker = np.zeros((1,))
        test_ticker = np.zeros((1,))
        for i in range(vocab_size):     #for each ticker, test the validation ticker against each other stock 
            validation_ticker[0,] = valid_ex_id
            test_ticker[0,] = i
            out = validation_model.predict([validation_ticker, test_ticker])
            sim[i] = out
        return sim

sim_cb = SimilarityCallback() #initialize class example

#**Training
arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))
for cnt in range(epochs):
    idx = np.random.randint(0, len(labels)-1)
    arr_1[0,] = targets[idx]
    arr_2[0,] = contexts[idx]
    arr_3[0,] = labels[idx]
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
    if cnt % 1000 == 0:
        print(f'Iteration {cnt}, loss={loss}')
#    if cnt % 10000 == 0:
#        sim_cb.run_sim()
    if cnt % 100000 == 0:
#        model.save(f'/gdrive/MyDrive/210422ETF2Vecweights{cnt}.tf')
        model.save(f'/Constellation_saved_weights/210422ETF2Vecweights{cnt}.tf')

