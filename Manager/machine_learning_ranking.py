from classifiers import KFoldCrossValidation
from utils import get_labels, get_rankings, list_split, get_ml_rankings
import numpy as np

class MLRanking(object):

    def __init__(self, corpus, method, classifier, kfold, nodeManager):
        self.corpus = corpus
        self.method = method
        self.classifier = classifier
        self.kfold = kfold
        self.nodeManager = nodeManager

    def rank_by_machine_learning(self):
        if self.method == 'method1':
            return self.first_method()
        else:
            return self.second_method()


    def first_method(self):
        print 'First method!'
        all_documentRankings = self.nodeManager.get_network_features()
        all_features = []
        all_labels = []
        doc_lenghts = []
        doc_names = []

        for i in all_documentRankings: #######   Modificar para MAchine learning com multiples tipos de redes complejas (segun limiares)      Por ahora solo se utilizara como se fuese una sola red
            # allRankings_for_doc_i = all_documentRankings[i] # lista de diccionarios por cada red compleja (generadas por los limiares caso sea MLN o embeddings) / MODIFICAR LUEGO!!!
            doc_names.append(i)
            allRankings_for_doc_i = all_documentRankings[i][0]  # lista de diccionarios por cada red compleja (generadas por los limiares caso sea MLN o embeddings) / MODIFICAR LUEGO!!
            document_data_for_doc_i = self.corpus[i]
            document_labels = get_labels(document_data_for_doc_i[1])
            rankings = get_rankings(allRankings_for_doc_i)
            doc_lenghts.append(len(rankings))
            all_features.extend(rankings)
            all_labels.extend(document_labels)


        for i , j in zip(all_features, all_labels):
            print i, j

        print ''
        print 'waaaaaaa'


        all_features = np.array(all_features)
        all_labels = np.array(all_labels)




        obj = KFoldCrossValidation(all_features, all_labels, self.classifier)
        predictions = obj.train_and_predict()

        for i, j in zip(all_labels, predictions):
            print i , j

        a = input()


        partitions = list_split(predictions, doc_lenghts)
        document_rankings = get_ml_rankings(doc_names, partitions)
        return document_rankings



    def second_method(self):
        # 'method2' usando machine learning con tradictional sumarization features  - no usar features de CN
        print 'Second method!'
        return 'En construccion ...'



if __name__ == '__main__':

    print 'hello Andoni'

    #obj = MLRanking()
    #obj.rank_by_ml()