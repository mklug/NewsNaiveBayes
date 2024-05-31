import math
import string
import collections
import numpy as np
import scipy.sparse


class BNB(object):
    '''
    BNB - Bernoulli Naive Bayes.
    Implementation of a Bernoulli naive Bayes classifier 
    using a sparse matrix representation of the parameter
    matrix and sparse matrix-vector multiplication for 
    prediction.  
    '''

    def __init__(self, alpha=1.0,
                 break_chars=set([' ', '\n'] + list(string.punctuation))):
        '''
        ``vocabulary`` : Dictionary that includes all of the words
                         in the training set after training. The 
                         nodes of the trie give indices where those 
                         words correspond to in matrix ``Theta``.

        ``Theta`` : Sparse matrix containing the probability
                    parameters computed during training.  This
                    is constructed initially as a COO sparse matrix
                    and then at the end of training converted to
                    a CSR matrix in anticipation of prediction
                    which involves computing matrix-vector products.

        ``log_priors`` : An array containing the logs of the priors 
                         estimated from the training data.

        ``alpha`` : Smoothing parameter applied at prediction time.  

        ``index_to_class`` : Assigns the indices of the resulting 
                             vector after prediction to classes.  
                             Used to output appropriate labels 
                             in prediction.  

        ``num_classes`` : Number of classes in the training data.

        ``num_words`` : Number of words in the training data.  

        ``class_index_to_word_count`` : Class index as in 
                                        ``index_to_class`` mapped
                                        to the number of words in 
                                        the class in the training
                                        data.  Used for smoothing
                                        in prediction.  

        ``break_chars`` : Set of characters that the input strings 
                          should be broken at.  
        '''
        self.vocabulary = {}
        self.Theta = None
        self.log_priors = None
        self.alpha = alpha
        self.index_to_class = {}
        self.num_classes = 0
        self.num_words = 0
        self.class_index_to_word_count = collections.Counter()
        self.break_chars = set([' ', '\n'] + list(string.punctuation))

    def _dict_process_train(self, x):
        '''Updates the vocabulary dictionary with the words in ``x``.
        Words are the result of breaking the string ``x`` at the 
        ``break_chars``.  The array of indices in the vocabulary 
        of this sequence of words is returned.  
        '''
        res = []
        current = []
        for c in x:
            if c in self.break_chars:
                current_word = ''.join(current)
                if current_word not in self.vocabulary:
                    self.vocabulary[current_word] = self.num_words
                    self.num_words += 1
                res.append(self.vocabulary[current_word])
                current = []
            else:
                current.append(c)
        return res

    def fit(self, X_train, y_train):
        '''Takes as input a sequence of training data and an 
        accompanying sequance of classes.  Fits the attributes
        using this data.  The ``Theta`` matrix is first constructed
        as a COO sparse matrix (note we utilize the option of 
        repeating entries in COO matrix initialization in scipy
        which results in all of the entries for a fixed coordinate
        being summed).  It is then converted to a CSR representation
        in anticipation of the matrix-vector multiplications 
        involved in prediction.  
        '''
        class_to_index = {}
        class_to_count = collections.Counter()

        data = np.array([], dtype=int)
        row = np.array([], dtype=int)
        col = np.array([], dtype=int)

        for x, y in zip(X_train, y_train):

            class_to_count[y] += 1
            if y not in class_to_index:
                class_to_index[y] = self.num_classes
                self.num_classes += 1

            x_processed = BNB._dict_process_train(self, x)
            self.class_index_to_word_count[class_to_index[y]
                                           ] += len(x_processed)

            N = len(x_processed)
            data = np.append(data, np.full(N, 1))
            row = np.append(row, np.full(N, class_to_index[y]))
            col = np.append(col, x_processed)

        num_words = len(self.vocabulary)
        self.index_to_class = {v: k for k, v in class_to_index.items()}
        self.log_priors = np.array([math.log(class_to_count[self.index_to_class[i]] / len(X_train))
                                    for i in range(self.num_classes)], dtype=float)

        self.Theta = scipy.sparse.coo_matrix((data, (row, col)),
                                             shape=(self.num_classes,
                                                    self.num_words),
                                             dtype=float).tocsr()

        for i, j in zip(self.Theta.nonzero()[0],
                        self.Theta.nonzero()[1]):
            self.Theta[i, j] = math.log((self.Theta[i, j] + self.alpha) /
                                        (self.class_index_to_word_count[i]
                                         + self.alpha * num_words))

    def _dict_process_test(self, x):
        '''Vectorizes the input x and returns the resulting 
        0,1 DOK sparse array.  
        '''
        res = scipy.sparse.dok_array((self.num_words, 1))
        current = []
        for c in x:
            if c in self.break_chars:
                current_word = ''.join(current)
                if current_word in self.vocabulary:
                    res[self.vocabulary[current_word], 0] = 1
                current = []
            else:
                current.append(c)
        return res

    def _get_max_index(nums):
        '''Returns the index of ``nums`` with the max value.'''
        return max(range(len(nums)), key=nums.__getitem__)

    def predict(self, X_test):
        '''Returns an array of class predictions.  
        '''

        res = []
        corrections = [math.log(self.alpha /
                                (self.class_index_to_word_count[i] +
                                 self.alpha * self.num_words))
                       for i in range(self.num_classes)]

        for x in X_test:
            x = BNB._dict_process_test(self, x)
            p0 = self.Theta.dot(x).toarray()

            # alpha correction.
            for i in range(self.num_classes):
                row = self.Theta[i, :]
                row_nz_row_indices = set(row.nonzero()[1])

                for index in x.nonzero()[0]:  # just row indices.
                    if index not in row_nz_row_indices:
                        p0[i, 0] += corrections[i]

            p0 = p0.T
            p0 += self.log_priors
            p0 = p0.reshape(self.num_classes)
            res.append(self.index_to_class[BNB._get_max_index(p0)])

        return res
