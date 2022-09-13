from gensim import corpora, models, similarities , matutils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from random import shuffle
from scipy import spatial
from utils import permutate_data , load_data_from_disk, get_w2v_vector, save_sentences, save_sentences_v2 ,get_fast_test_vector, get_fast_test_vector_v2
from utils import save_processed_sentences, save_processed_sentences_v2, get_fast_test_vector_s2v, extract_sentences
from utils import get_glove_matrix
from configuration import extras, some_parameters
import os
#import fasttext

from glove import Corpus, Glove




class Vectorization(object):

    #def __init__(self, corpus, vectorization_type, use_inference=None, vector_size=None, auxiliar_corpus=None):
    def __init__(self, corpus, vectorization_type, vector_size=None, auxiliar_corpus=None, language='ptg'):
        self.corpus = corpus
        self.vectorization_type = vectorization_type
        #self.use_inference = use_inference
        self.vector_size = vector_size
        self.auxiliar_corpus = auxiliar_corpus
        self.language = language

    def tf_idf_vectorization(self):
        obj = TfIdfModel(self.corpus, self.auxiliar_corpus)
        obj.train()
        return obj.get_matrix_tfidf()

    def d2v_vectorization(self):
        #obj = Doc2VecModel(self.corpus, self.use_inference, self.vector_size, self.auxiliar_corpus)
        obj = Doc2VecModel(self.corpus, self.vector_size, self.auxiliar_corpus)
        obj.train()
        return obj.get_matrix_doc2vec()

    def d2v_google_vectorization(self):
        print 'd2v google!!!'
        obj = Doc2VecModelGoogleNews(self.corpus)
        return obj.get_matrix_doc2vec_google()

    def fast_text_vectorization(self):
        print 'fast text vectorization!'
        obj = FastText(self.corpus, self.auxiliar_corpus, language=self.language)
        obj.train()
        return obj.get_matrix_fast_text()

    def sent2vec_vectorization(self):
        print 'sent2vec vectorization!'
        obj = Sent2Vec(self.corpus, self.auxiliar_corpus, language=self.language)
        obj.train()
        return obj.get_matrix_sent2vec()


    def glove_vectorization(self):
        print 'glove vectorization!'
        obj = GloveVectorization(self.corpus, self.auxiliar_corpus)
        obj.train()
        return obj.get_matrix_glove()


    def calculate(self):
        if self.vectorization_type == 'tfidf':
            return self.tf_idf_vectorization()
        elif self.vectorization_type == 'gd2v':
            return self.d2v_google_vectorization()
        elif self.vectorization_type == 'fastT':
            return self.fast_text_vectorization()
        elif self.vectorization_type == 'gloVe':
            return self.glove_vectorization()
        elif self.vectorization_type == 's2v':
            return self.sent2vec_vectorization()
        else:
            return self.d2v_vectorization()

        #return ['dictionary', 'key: nombre del documento o cluster', 'value: matrix con los vectores de cada sentence del documento']


class TfIdfModel(object):

    def __init__(self, corpus, auxiliar=None):
        print "vectorization tfidf!!"
        self.corpus = corpus
        self.auxiliar = auxiliar
    '''
            for i in processed_corpus.items():
            valores =  i[1]
            psentences = valores[1]
            for j in psentences:
                print j[0] , j[1]
    '''

    def train(self):
        allSentences = []
        for i in self.corpus.items():
            sentence_values = i[1]
            psentences = sentence_values[1]
            for j in psentences:
                allSentences.append(j[0])


        if self.auxiliar is not None:
            for i in self.auxiliar.items():
                sentence_values = i[1]
                psentences = sentence_values[1]
                for j in psentences:
                    allSentences.append(j[0])

        self.dictionary = corpora.Dictionary(allSentences)
        theCorpus = [self.dictionary.doc2bow(text) for text in allSentences]
        self.tfidf = models.TfidfModel(theCorpus)


    def get_matrix_tfidf(self):
        corpus_matrix = dict()
        for i in self.corpus.items():
            doc_name = i[0]
            doc_sentences = i[1][1]
            doc_matrix = []
            for j in doc_sentences:
                vec_bow = self.dictionary.doc2bow(j[0])  # modificar aquii,, preprocesed (content  , id=null)
                vec_tfidf = self.tfidf[vec_bow]
                doc_matrix.append(vec_tfidf)

            corpus_matrix[doc_name] = doc_matrix
        return corpus_matrix


class Doc2VecModel(object):

    #def __init__(self, corpus, inference, size, auxiliar):
    def __init__(self, corpus, size, auxiliar):
        print "vectorizacion doc2vec!!"
        self.corpus = corpus
        #self.inference = inference
        self.size = size
        self.auxiliar = auxiliar

    def train(self):
        allSentences = []
        for i in self.corpus.items():
            doc_name = i[0]
            sentences = i[1][1]     # modificar aquiii!!
            for index, sent in enumerate(sentences):
                sent_name = doc_name + "_" + str(index)
                allSentences.append((sent[0], sent_name))

        if self.auxiliar is not None:
            for i in self.auxiliar.items():
                doc_name = i[0]
                sentences = i[1][1]     # modificar aqui?
                for index, sent in enumerate(sentences):
                    sent_name = doc_name + "_" + str(index)
                    allSentences.append((sent[0], sent_name))

        labeled_sentences = []
        #if self.inference:
        #    print "aun falta implementarrr!"
        #    print "posible error aqui!"

        for i in allSentences:
            sentence = i[0]
            label = i[1]
            labeled_sentences.append(LabeledSentence(sentence, [label]))

        #self.model = Doc2Vec(min_count=1, window=10, size=self.size, sample=1e-4, negative=5, workers=8)
        #self.model = Doc2Vec(min_count=1, window=10, size=self.size, sample=1e-4, workers=8)
        self.model = Doc2Vec(min_count=1, window=10, size=self.size, workers=8)  # okk
        self.model.build_vocab(labeled_sentences)
        print "training d2v ...."
        #for epoch in range(10):
        #    self.model.train(permutate_data(labeled_sentences), total_examples=self.model.corpus_count)
        self.model.train(labeled_sentences, total_examples=self.model.corpus_count, epochs=self.model.iter)
        #self.model.train(labeled_sentences, total_examples=self.model.corpus_count, epochs=10)




    def get_matrix_doc2vec(self):
        print "obtaining matrix"
        corpus_matrix = dict()
        for i in self.corpus.items():
            doc_name = i[0]
            size = len(i[1][1])
            doc_matrix = []
            for i in range(size):
                key = doc_name + "_" + str(i)
                vec_d2v = self.model.docvecs[key]
                doc_matrix.append(vec_d2v)
            corpus_matrix[doc_name] = doc_matrix

        return corpus_matrix


class Doc2VecModelGoogleNews(object):

    def __init__(self, corpus):
        self.corpus = corpus
        self.w2v_vocabulary = load_data_from_disk(extras['google_w2v'])

    def get_matrix_doc2vec_google(self):
        corpus_matrix = dict()
        for i in self.corpus.items():
            doc_name = i[0]
            sentences = i[1][1]
            doc_matrix = []
            for index, sentence in enumerate(sentences):
                sentence_w2v_vector = get_w2v_vector(self.w2v_vocabulary  ,sentence[0])
                doc_matrix.append(sentence_w2v_vector)
            corpus_matrix[doc_name] = doc_matrix
        return corpus_matrix



class FastText(object):

    # type='cbow' \ 'skipgram'    proccessing: true->utilizar la parte preprocesada  false-> usar las sentencias originales
    def __init__(self, corpus, auxiliar=None, type='cbow', use_proccessing=False, use_pre_trained=True, language='ptg'):
        self.corpus = corpus
        self.auxiliar = auxiliar
        self.type = type
        self.proccessing = use_proccessing
        self.auxiliar_sentence_list = None
        self.use_pre_trained_vectors = use_pre_trained
        self.language = language

        allSentences = []
        pAllSentences = []

        for i in self.corpus.items():
            #print i[0]
            original_sentences =  i[1][0]
            preprocesed_sentences = i[1][1]
            allSentences.append(original_sentences)
            pAllSentences.append(preprocesed_sentences)


        if self.auxiliar is not None:
            for i in self.auxiliar.items():
                original_sentences = i[1][0]
                preprocesed_sentences = i[1][1]
                allSentences.append(original_sentences)
                pAllSentences.append(preprocesed_sentences)

        if self.proccessing: # use preprocesed
            self.auxiliar_sentence_list = save_processed_sentences(some_parameters['train_file'], allSentences, pAllSentences)
            # list of all sentences from Temario o CSTNews ,,,, o de duc2002  - duc2004

        else: # use original
            save_sentences(some_parameters['train_file'], allSentences)


        #self.generate_train_command = './fasttext ' + self.type + ' -input ' + some_parameters['train_file'] + ' -output model'
        self.generate_train_command = './fasttext ' + self.type + ' -input ' + some_parameters['train_file'] + ' -output ' + some_parameters['model']
        print self.generate_train_command

    def train(self):
        print 'training ...'
        if self.use_pre_trained_vectors:
            pass
        else:
            print "habeeeeer"
            os.system(self.generate_train_command)
            print self.generate_train_command

            #a = input()


    def get_matrix_fast_text(self):

        corpus_matrix = dict()
        allSentences = []
        for i in self.corpus.items():
            doc_sentences = i[1][0]
            allSentences.extend(doc_sentences) ############


        if self.proccessing: # use preprocesed
            pSentences = save_processed_sentences_v2(some_parameters['test_file'], allSentences, self.auxiliar_sentence_list)
        else: # use original
            pSentences = save_sentences_v2(some_parameters['test_file'], allSentences)  ###################


        vectors_dictionary = get_fast_test_vector(allSentences, pSentences, some_parameters['test_file'], use_pre_trained=self.use_pre_trained_vectors, language=self.language)  ############

        for i in self.corpus.items():
            doc_name = i[0]
            doc_sentences = i[1][0]
            doc_matrix = []
            for j in doc_sentences:
                vector = vectors_dictionary[j]
                doc_matrix.append(vector)
            corpus_matrix[doc_name] = doc_matrix


        return corpus_matrix


    def get_matrix_fast_text_v2(self):
        corpus_matrix = dict()
        model = fasttext.load_model('model.bin')

        for i in self.corpus.items():
            doc_name = i[0]
            doc_sentences = i[1][0]
            doc_matrix = []
            for j in doc_sentences:
                vector = get_fast_test_vector_v2(model, j)
                doc_matrix.append(vector)
            corpus_matrix[doc_name] = doc_matrix
        return corpus_matrix



class Sent2Vec(object):

    def __init__(self, corpus, auxiliar=None, use_proccessing=False, use_pre_trained=False, language='ptg'):
        self.corpus = corpus
        self.auxiliar = auxiliar
        self.proccessing = use_proccessing
        self.auxiliar_sentence_list = None
        self.use_pre_trained_vectors = use_pre_trained
        self.language = language

        allSentences = []
        pAllSentences = []

        for i in self.corpus.items():
            original_sentences =  i[1][0]
            preprocesed_sentences = i[1][1]
            allSentences.append(original_sentences)
            pAllSentences.append(preprocesed_sentences)


        if self.auxiliar is not None:
            for i in self.auxiliar.items():
                original_sentences = i[1][0]
                preprocesed_sentences = i[1][1]
                allSentences.append(original_sentences)
                pAllSentences.append(preprocesed_sentences)

        if self.proccessing:  # use preprocesed
            self.auxiliar_sentence_list = save_processed_sentences(some_parameters['train_file_v2'], allSentences, pAllSentences) # list of all sentences from Temario o CSTNews ,,,, o de duc2002  - duc2004
        else:  # use original
            save_sentences(some_parameters['train_file_v2'], allSentences)



        self.generate_train_command = './fasttext2 sent2vec -input ' + some_parameters['train_file_v2'] + ' -output ' + some_parameters['model_v2']
        print self.generate_train_command


    def train(self):
        print 'training ...'
        if self.use_pre_trained_vectors:
            pass
        else:
            os.system(self.generate_train_command) ##########################################################################################

    def get_matrix_sent2vec(self):
        corpus_matrix = dict()
        allSentences = []
        for i in self.corpus.items():
            doc_sentences = i[1][0]
            allSentences.extend(doc_sentences)

        if self.proccessing: # use preprocesed
            save_processed_sentences_v2(some_parameters['test_file_v2'], allSentences, self.auxiliar_sentence_list)
        else: # use original
            save_sentences_v2(some_parameters['test_file_v2'], allSentences)  ###################


        vectors_dictionary = get_fast_test_vector_s2v(allSentences, some_parameters['test_file_v2'])  ############


        for i in self.corpus.items():
            doc_name = i[0]
            doc_sentences = i[1][0]
            doc_matrix = []
            for j in doc_sentences:
                vector = vectors_dictionary[j]
                doc_matrix.append(vector)
            corpus_matrix[doc_name] = doc_matrix

        return corpus_matrix


'''
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
            ['this', 'is', 'the', 'second', 'sentence'],
            ['yet', 'another', 'sentence'],
            ['one', 'more', 'sentence'],
            ['and', 'the', 'final', 'sentence']]


corpus = Corpus()
corpus.fit(sentences, window=10)
glove = Glove(no_components=100, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)
glove.save('glove.model')
glove = Glove.load('glove.model')
vectors = glove.word_vectors
index = glove.dictionary['and']
print vectors[index]
'''

class GloveVectorization(object):

    def __init__(self, corpus, auxiliar=None, use_proccessing=False):
        self.corpus = corpus
        self.auxiliar = auxiliar
        self.proccessing = use_proccessing

        self.allSentences = []
        self.pAllSentences = []

        for i in self.corpus.items():
            original_sentences =  i[1][0]
            preprocesed_sentences = i[1][1]

            self.allSentences.extend(extract_sentences(original_sentences, False))
            self.pAllSentences.extend(extract_sentences(preprocesed_sentences, True))



        if self.auxiliar is not None:
            for i in self.auxiliar.items():
                original_sentences = i[1][0]
                preprocesed_sentences = i[1][1]

                self.allSentences.extend(extract_sentences(original_sentences, False))
                self.pAllSentences.extend(extract_sentences(preprocesed_sentences, True))


    def train(self):
        print 'training glove ...'
        corpus = Corpus()
        if self.proccessing:
            corpus.fit(self.pAllSentences, window=10)
        else:
            corpus.fit(self.allSentences, window=10)

        model = Glove(no_components=300, learning_rate=0.05)
        model.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
        model.add_dictionary(corpus.dictionary)
        model.save(some_parameters['glove_model']) #
        

    def get_matrix_glove(self):
        model = Glove.load(some_parameters['glove_model'])
        vectors = model.word_vectors
        dictionary = model.dictionary
        corpus_matrix = dict()

        for i in self.corpus.items():
            doc_name = i[0]
            original_sentences =  i[1][0]
            preprocesed_sentences = i[1][1]

            ori_sents = extract_sentences(original_sentences, False)
            pp_sents = extract_sentences(preprocesed_sentences, True)


            if self.proccessing:
                matrix = get_glove_matrix(pp_sents, dictionary, vectors) 
            else:
                matrix = get_glove_matrix(ori_sents, dictionary, vectors)


            corpus_matrix[doc_name] = matrix

        return corpus_matrix 
        