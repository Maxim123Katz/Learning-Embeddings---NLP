"""
API for ex2, implementing the skip-gram model (with negative sampling).

"""

# you can use these packages (uncomment as needed)
import pickle
# import pandas as pd
import numpy as np
import os, time, re, sys, random, math, collections, nltk
from collections import Counter
from nltk import skipgrams, sent_tokenize


# static functions
def who_am_i():  # this is not a class method
    """Returns a dictionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Maxim Katz', 'id': '322406604', 'email': 'katzmax@post.bgu.ac.il'}


def normalize_text(fn):
    """ Loading a text file and normalizing it, returning a list of sentences.

    Args:
        fn: full path to the text file to process
    """
    with open(fn, "r") as file:
        text = file.read() # read the entire file

    lines = sent_tokenize(text) # split the text into sentences
    sentences = [
        re.sub(r'["“”.,!?|]+', "", line).replace("'", "").lower() # remove punctuation and lower case
        for line in lines
        if line and line.strip() # remove empty lines
    ]

    return sentences # return a list of sentences


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def load_model(fn):
    """ Loads a model pickle and return it.
    Args:
        fn: the full path to the model to load.
    """
    with open(fn, 'rb') as f: # open the file for reading
        sg_model = pickle.load(f) # load the model
    return sg_model # return the model


class SkipGram:
    def __init__(self, sentences, d=100, neg_samples=4, context=4, word_count_threshold=5):
        self.sentences = sentences # list of sentences
        self.d = d  # embedding dimension
        self.neg_samples = neg_samples  # num of negative samples for one positive sample
        self.context = context  # the size of the context window (not counting the target word)
        self.word_count_threshold = word_count_threshold  # ignore low frequency words (appearing under the threshold)

        # Tips:
        # 1. It is recommended to create a word:count dictionary
        # 2. It is recommended to create a word-index map

        self.T = None  # Target matrix embeddings
        self.C = None  # Context matrix embeddings
        self.V = None  # Combined matrix embeddings
        self.pos_context_neg = [] # list of dictionaries {target: (context, negative samples)}
        self.word_counts = Counter() # {word:count dictionary}
        # tip 1:
        word_list = []  # list of all words in the text
        for s in sentences: # split the sentences into words
            word_list.extend(s.split(" ")) # split the sentence into words
        self.word_counts = Counter(word_list) # count the words

        # tip 2:
        self.w2i = {} # word to index
        self.i2w = {} # index to word
        self.idx = 0 # vocabulary size
        remove = []
        for w, c in self.word_counts.items(): # create the word to index and index to word dictionaries
            if c >= word_count_threshold: # ignore low frequency words
                self.w2i[w] = self.idx # add the word to the word to index dictionary
                self.i2w[self.idx] = w # add the index to word dictionary
                self.idx += 1 # increase the vocabulary size
            else: # remove low frequency words
                remove.append(w)  # add the word to the remove list

        for w in remove: # remove the low frequency words from the word counts
            del self.word_counts[w] # remove the word from the word counts

        # additional:
        self.words_sum = sum(self.word_counts.values()) # sum of all words in the text
        self.word_list_ord = [self.i2w[i] for i in range(self.idx)]
        self.word_prob = []
        for w in self.word_list_ord:
            self.word_prob.append(self.word_counts[w] / self.words_sum)
        self.word_prob = np.array(self.word_prob)

        self.sentences_as_idx = [  # list of sentences, each sentence is a list of word indices
            [self.w2i[w] for w in s.split() if w in self.w2i]
            for s in self.sentences]

    def posneg_samples(self):
        for s in self.sentences: # iterate over the sentences
            tokens = [word for word in s.split() if word in self.word_counts] # split the sentence into words

            pos_samples = list(skipgrams(tokens, 2, self.context // 2 - 1)) + \
                          list(skipgrams(tokens[::-1], 2, self.context // 2 - 1)) # create positive samples

            con_tar_dic = {} # dictionary to store the positive samples
            for target, context in pos_samples: # iterate over the positive samples
                target_idx = self.w2i[target] # get the target index
                context_idx = self.w2i[context] # get the context index
                if target_idx not in con_tar_dic: # add the target to the dictionary
                    con_tar_dic[target_idx] = [] # create a list for the target
                con_tar_dic[target_idx].append(context_idx) # add the context to the target list

            if con_tar_dic: # add the positive samples to the list
                self.pos_context_neg.append(con_tar_dic) # add the positive samples to the list

        for dic in self.pos_context_neg: # iterate over the positive samples
            for target_idx, context_indices in dic.items(): # iterate over the positive samples
                negative_samples = random.choices(list(self.word_counts.keys()),
                                                  weights=list(self.word_counts.values()),
                                                  k=self.neg_samples * len(context_indices)) # create negative samples
                negative_samples = [self.w2i[c] for c in negative_samples if c != target_idx] # get the negative samples
                dic[target_idx] = (context_indices, negative_samples) # add the negative samples to the dictionary
        return self.pos_context_neg # return the list of dictionaries

    def compute_similarity(self, w1, w2):
        """ Returns the cosine similarity (in [0,1]) between the specified words.
        Args:
            w1: a word
            w2: a word
        Retunrns: a float in [0,1]; defaults to 0.0 if one of specified words is OOV.
        """
        sim = 0.0  # default
        if self.V is None:
            return sim
        if w1 in self.w2i and w2 in self.w2i:
            vec_w1 = self.T[:, self.w2i[w1]]
            vec_w2 = self.T[:, self.w2i[w2]]
            scale_w1 = np.linalg.norm(vec_w1)
            scale_w2 = np.linalg.norm(vec_w2)
            if scale_w1 <= 0 or scale_w2 <= 0:
                return sim
            sim = np.dot(vec_w1, vec_w2) / (scale_w1 * scale_w2)
        return sim

    def get_closest_words(self, w, n=5):
        """Returns a list containing the n words that are the closest to the specified word.

        Args:
            w: the word to find close words to.
            n: the number of words to return. Defaults to 5.
        """
        if not w or w not in self.w2i:
            return []  # if the word is not in the vocabulary or is empty
        # Create a list of (word, similarity) tuples, excluding the target word itself
        word_sim = [] # List to store (word, similarity) tuples
        for word in self.w2i.keys(): # Iterate over all words in the vocabulary
            if word != w: # Exclude the target word
                cos_sim = self.compute_similarity(w, word) # Compute the cosine similarity
                word_sim.append((word, cos_sim)) # Add the word and similarity to the list
        # Sort the list by similarity in descending order
        word_sim.sort(key=lambda x: x[1], reverse=True)
        # Return the top n words
        return [word for word, sim in word_sim[:n]]

    def learn_embeddings(self, step_size=0.001, epochs=50, early_stopping=3, model_path=None):
        """Returns a trained embedding models and saves it in the specified path

        Args:
            step_size: step size for  the gradient descent. Defaults to 0.0001
            epochs: number or training epochs. Defaults to 50
            early_stopping: stop training if the Loss was not improved for this number of epochs
            model_path: full path (including file name) to save the model pickle at.
        """

        vocab_size = self.idx
        T = np.random.rand(self.d, vocab_size)  # embedding matrix of target words
        C = np.random.rand(vocab_size, self.d)  # embedding matrix of context words

        # tips:
        # 1. have a flag that allows printing to standard output so you can follow timing, loss change etc.
        # 2. print progress indicators every N (hundreds? thousands? an epoch?) samples
        # 3. save a temp model after every epoch
        # 4.1 before you start - have the training examples ready - both positive and negative samples
        # 4.2. it is recommended to train on word indices and not the strings themselves.

        pos_neg = self.posneg_samples() # list of dictionaries {target: (context, negative samples)}
        flag = False
        if flag:
            print("the preprocessing finish")
        delta = 10 ^ (-3)
        loss_lst = []  # List to store loss for each epoch
        early_stop_count = 0 # Counter for early stopping
        for i in range(1, epochs + 1): # iterate over the epochs
            random.shuffle(pos_neg) # shuffle the positive and negative samples
            epoch_loss = [] # List to store loss for each epoch
            for spn in pos_neg: # iterate over the positive and negative samples
                for t_index, (pos_sample, neg_sample) in spn.items(): # iterate over the positive and negative samples
                    neg_y = np.zeros(len(neg_sample), dtype=int)  # False labels for negative samples
                    pos_y = np.ones(len(pos_sample), dtype=int)  # True labels for positive samples
                    y_true = np.concatenate((pos_y, neg_y)).reshape((-1, 1)) # Concatenate the labels
                    s_neg_pos = pos_sample + neg_sample # Concatenate the positive and negative samples

                    # Forward step:
                    target_embedding = T[:, t_index].reshape(-1, 1) # Get the target embedding
                    pos_neg_matrix = C[s_neg_pos] # Get the context and negative samples embeddings
                    output_layer = np.dot(pos_neg_matrix, target_embedding) # Calculate the output layer
                    y_pred = sigmoid(output_layer) # Calculate the predicted labels

                    # Loss calculation:
                    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # Clip the predicted labels
                    loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)) # Calculate the loss
                    epoch_loss.append(loss) # Add the loss to the epoch loss
                    error = y_pred - y_true # Calculate the error

                    # Gradient calculation:
                    T_grad = np.dot(pos_neg_matrix.T, error) # Calculate the gradient for the target embedding
                    C_grad = np.dot(error, target_embedding.T) # Calculate the gradient for the context and negative samples embeddings

                    # Update embeddings:
                    T[:, [t_index]] -= step_size * T_grad # Update the target embedding
                    C[s_neg_pos] -= step_size * C_grad # Update the context and negative samples embeddings
            # update the T and C metrics embeddings
            self.T = T
            self.C = C
            avg_loss_epoch = np.mean(epoch_loss)
            if flag:
                print(f"Epoch {i}/{epochs} - Loss: {avg_loss_epoch}")
            loss_lst.append(avg_loss_epoch)
            # After each epoch check the early stopping: stop training if the Loss was not improved
            if len(loss_lst) > 1:
                if abs(loss_lst[-1] - loss_lst[-2]) < delta: # Check if the loss was not improved
                    early_stop_count += 1
                    if early_stop_count >= early_stopping: # Check if the early stopping condition is met
                        if flag:
                            print(f"Early Stopping")
                        break
                else:
                    early_stop_count = 0

        # save the model:
        if model_path is not None:
            with open(model_path, "wb") as f:
                pickle.dump(self, f)
        self.V = self.combine_vectors(T, C, 4, model_path) # Combine the T and C embeddings
        return T, C

    def combine_vectors(self, T, C, combo=0, model_path=None):
        """Returns a single embedding matrix and saves it to the specified path

        Args:
            T: The learned targets (T) embeddings (as returned from learn_embeddings())
            C: The learned contexts (C) embeddings (as returned from learn_embeddings())
            combo: indicates how wo combine the T and C embeddings (int)
                   0: use only the T embeddings (default)
                   1: use only the C embeddings
                   2: return a pointwise average of C and T
                   3: return the sum of C and T
                   4: concat C and T vectors (effectively doubling the dimention of the embedding space)
            model_path: full path (including file name) to save the model pickle at.
        """

        V = None
        if combo == 0: # use only the T embeddings
            V = T
        elif combo == 1: # use only the C embeddings
            V = C.T
        elif combo == 2: # return a pointwise average of C and T
            V = (T + C.T) / 2
        elif combo == 3: # return the sum of C and T
            V = T + C.T
        elif combo == 4: # concat C and T vectors
            V = np.concatenate((T.T, C), axis=1).T

        if model_path is not None: # save the model
            with open(model_path, "wb") as f:
                pickle.dump(V, f)
        self.V = V # set the V attribute
        return V

    def find_analogy(self, w1, w2, w3):
        """Returns a word (string) that matches the analogy test given the three specified words.
           Required analogy: w1 to w2 is like ____ to w3.

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
        """

        if any(word not in self.w2i for word in (w1, w2, w3)): # if any of the words is OOV
            return '' # return an empty string

        w1_idx = self.w2i[w1] # Get the word index
        w2_idx = self.w2i[w2] # Get the word index
        w3_idx = self.w2i[w3] # Get the word index

        w1_vector = self.V[:, w1_idx] # Get the word vectors
        w2_vector = self.V[:, w2_idx] # Get the word vectors
        w3_vector = self.V[:, w3_idx] # Get the word vectors

        analogy_vector = w1_vector - w2_vector + w3_vector # Calculate the analogy vector
        analogy_vector_norm = np.linalg.norm(analogy_vector) # Calculate the norm of the analogy vector
        T_norms = np.linalg.norm(self.V.T, axis=1) # Calculate the norms of the word vectors
        e = 1e-8 # Small number to avoid division by zero
        similarity_scores = np.dot(self.V.T, analogy_vector) / (T_norms * analogy_vector_norm + e) # Calculate the cosine similarity
        sort_similar_id = np.argsort(similarity_scores)[::-1]  # Sort by cosine similarity key
        top_w = [self.i2w[index] for index in sort_similar_id][:5]  # closest word index

        for wor in top_w: # iterate over the closest words
            if wor not in {w1, w2, w3}: # if the word is not one of the words in the analogy
                return wor # return the word

        return wor # return the word

    def test_analogy(self, w1, w2, w3, w4, n=1):
        """Returns True if sim(w1-w2+w3, w4)@n; Otherwise return False.
            That is, returning True if w4 is one of the n closest words to the vector w1-w2+w3.
            Interpretation: 'w1 to w2 is like w4 to w3'

        Args:
             w1: first word in the analogy (string)
             w2: second word in the analogy (string)
             w3: third word in the analogy (string)
             w4: forth word in the analogy (string)
             n: the distance (work rank) to be accepted as similarity
            """

        targets = self.find_analogy(w1, w2, w3) # find the analogy
        top_n_words = self.get_closest_words(targets, n=n) # get the closest words
        if w4 in top_n_words or targets == w4: # if the forth word is in the top n words
            return True
        return False

