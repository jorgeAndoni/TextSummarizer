from corpus_loader import Loader
from utils import parameter_extractor, deleteFolders, get_labels, get_rankings, list_split
from embeddings_generator import Vectorization
from text_conversion import CorpusConversion
from network import NetworkManager , NodeManager
from summarization import SummaryGenerator
from validation import Validation
from configuration import extras, final_results
from random import shuffle, choice
from classifiers import KFoldCrossValidation
import numpy as np
from machine_learning_ranking import MLRanking

class Summarizer(object):
    def __init__(self, test, output):
        self.data = self.parse_file(test)
        self.output_excel = output


    def execute(self):
        data = self.data
        language = data['language']
        if data['type'][0] == 'SDS':
            type_summary = 0
        else:
            type_summary = 1
        anti_redundancy_method = data['type'][1]

        corpus_name = data['corpus']
        #resumo_size_parameter = data['size']  # para definir el tamanio de los sumarios, en relacion a numero de palabras o sentencias, o fijo

        use_machine_learning = data['ml'][0]  ## VERY IMPORTANT NOW
        method, classifier, kFold, use_traditional_features  = None , None, None, None
        if use_machine_learning:
            method, classifier, kFold, use_traditional_features = data['ml'][1][0], data['ml'][1][1], data['ml'][1][2], data['ml'][1][3]


        network = data['network']
        network_type = network[0]  # tipo de red: noun, tfidf, d2v , mln
        network_parameters = network[1]  # todos los parametros del tipo de red que se va a utilizar
        mln_type_flag = network_type == 'mln'  # para verificar en corpus loader si se tiene que cargar para una multilayer network

        extracted_net_parameters = parameter_extractor(network_type, network_parameters)

        mln_type = extracted_net_parameters['mln_type']
        sw_removal = extracted_net_parameters['sw_removal']
        limiar_value = extracted_net_parameters['limiar_value']
        limiar_type = extracted_net_parameters['limiar_type']
        size_d2v = extracted_net_parameters['size_d2v']
        inter_edge_mln = extracted_net_parameters['inter_edge']
        limiar_mln = extracted_net_parameters['limiar_mln']

        network_measures = data['measures']
        #selection_method = data['selection']  #####

        #print use_machine_learning
        #print method, classifier, kFold, use_traditional_features
        # use_machine_learning and method   ---> muy importantes


        print extracted_net_parameters


        '''
        1 Corpus loader : cargar el corpus indicado y dejarlo listo para ser pre-procesado    
        '''

        #obj = Loader(language=language, type_summary=type_summary, corpus=corpus_name, size=resumo_size_parameter, mln=mln_type_flag, use_ml=use_machine_learning)
        obj = Loader(language=language, type_summary=type_summary, corpus=corpus_name, mln=mln_type_flag, use_ml=use_machine_learning)
        loaded_corpus = obj.load()  # diccionario que tiene como key el nombre del documento o nombre del grupo y como claves los documentos y sus sizes



        '''
        2. Corpus processing
        '''
        obj = CorpusConversion(loaded_corpus, language, network_type, mln_type, sw_removal)
        processed_corpus = obj.convert()

        #for i in processed_corpus.items():
        #    print i


        '''
        3. Corpus vectorization 
        '''
        
        vectorized_corpus = None

        if network_type == 'noun' or mln_type == 'noun':
            pass


        else:
            if  network_type== 'mln':
                network_type_subtype = mln_type
            else:
                network_type_subtype = network_type


            if language == 'eng':
                obj = Vectorization(processed_corpus, network_type_subtype, size_d2v, language=language)
                vectorized_corpus = obj.calculate()
            else:
                type_summary_inverted = 0
                if type_summary == 0:
                    type_summary_inverted = 1

                obj = Loader(language=language, type_summary=type_summary_inverted, corpus=corpus_name, mln=mln_type_flag)
                auxiliar_corpus = obj.load()


                obj = CorpusConversion(auxiliar_corpus, language, network_type, mln_type, sw_removal)
                processed_auxiliar = obj.convert()

                obj = Vectorization(processed_corpus, network_type_subtype, size_d2v, processed_auxiliar, language=language)
                vectorized_corpus = obj.calculate()




        '''
        4. Network creation
        5. Network prunning 
        '''
        

        obj = NetworkManager(network_type, mln_type, processed_corpus, vectorized_corpus, inter_edge_mln, limiar_mln, limiar_value, limiar_type)
        complex_networks = obj.create_networks()


        '''
        6. Node weighting  7. Node ranking
        6. Node weighting 7. Machine Learning
        '''
        

        manageNodes = NodeManager(complex_networks, network_measures)

        #features = manageNodes.get_network_features()

        if use_machine_learning:
            obj = MLRanking(corpus=processed_corpus, method=method, classifier=classifier, kfold=kFold, nodeManager=manageNodes)
            all_documentRankings = obj.rank_by_machine_learning()
        else:
            all_documentRankings = manageNodes.ranking()

        



        #for i in all_documentRankings.items():
        #    print i




        '''
        8. Summarization
        '''
        
        
        obj = SummaryGenerator(processed_corpus, complex_networks, all_documentRankings,anti_redundancy_method)
        obj.generate_summaries()
        






        '''
         9. Validation
        '''
        
        
        key = choice(all_documentRankings.keys())
        number_of_measures = len(all_documentRankings[key][0])
        parameters_to_show_table = []

        if limiar_mln is not None:
            first_value = len(inter_edge_mln)
            second_value = len(limiar_mln)
            third_value = number_of_measures
            parameters_to_show_table.append(inter_edge_mln)
            parameters_to_show_table.append(limiar_mln)
        elif limiar_value is not None:
            first_value = 1
            second_value = len(limiar_value)
            third_value = number_of_measures
            parameters_to_show_table.append(None)
            parameters_to_show_table.append(limiar_value)
        else:
            first_value = 1
            second_value = 1
            third_value = number_of_measures

        print first_value, second_value, third_value

        obj = Validation(language, type_summary, corpus_name, [first_value, second_value, third_value], self.output_excel, parameters_to_show_table)
        obj.validate('results.csv')
        
        deleteFolders(extras['Automatics'])
        
             







    def parse_file(self, doc):
        dictionary = dict()
        #dictionary['language'] = 'ptg'
        dictionary['language'] = 'eng'
        #dictionary['type'] = ('SDS' , None)
        dictionary['type'] = ('MDS', 1)  # 0->sin antiredundancia, 1->metodo de ribaldo 2->metodo de ngrams  3-> maximum marginal relevance
        dictionary['corpus'] = 0  # 1  para DUC2004 en caso del ingles, solo para MDS
        #dictionary['size'] = 'w'  ## removeee!
        #dictionary['ml'] = (True, ['method1','naive_bayes' , 10, False])  # metodo ,classifier(naive_bayes/svm/decision_tree/logistic) , kfoldcrossvalidation (10), use traditional measures
        # method1 --> training and testing CN y ML
        # method2 --> primero hago ranking como estaba usando antes y luego para la seleccion final de sentencias aplico ml en el ranking final para verificar si la sentencia va o no va
        dictionary['ml'] = (False, [])
        '''
        - forma de usar ml:  1. usar enfoque training and testing (cn y ml) o  2. todo como la metodoligia anterior y al final usar ml para verificar los rankings
        - clasificador: naive bayes , svm, dt , logistic regression , etc
        - k fold cross validation : 10 
        - features CN : las que son seleccionadas en cn measures
        - features tradicionales: seleccionar features tradicionales
        - CN + tradicionales      
        '''



        #dictionary['network'] = ('noun', [])
        #dictionary['network'] = ('tfidf', [])
        #dictionary['network'] = ('fastT' , [('limiar', [0.15, 0.20, 0.25, 0.3, 0.35,0.40])])
        #dictionary['network'] = ('s2v', [('limiar', [0.15, 0.20, 0.25, 0.3, 0.35,0.40])])
        dictionary['network'] = ('gloVe', [('limiar', [0.15, 0.20, 0.25, 0.3, 0.35, 0.4])])
        #dictionary['network'] = ('d2v', [False, ('limiar', [0.15, 0.20, 0.25, 0.30, 0.35, 0.4]), 300])
        #dictionary['network'] = ('gd2v', [('limiar', [0.15, 0.20, 0.25, 0.3, 0.35, 0.4])])



        #dictionary['network'] = ('mln', ['tfidf', [1.1, 1.3, 1.5, 1.7, 1.9], [0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]]) # inter - limiar remocion
        # dictionary['network'] = ('mln', ['noun', [1.1, 1.3, 1.5], [0.1, 0.15, 0.20]])


        #dictionary['measures'] = ['dg' , 'pr', 'gaccs']
        #dictionary['measures'] = ['at' , 'gaccs']
        dictionary['measures'] = ['*']
        #dictionary['measures'] = ['gaccs']
        #dictionary['measures'] = ['sp' , 'pr' , 'btw' , 'cc']


        return dictionary


if __name__ == '__main__':

    output = final_results['prueba2']
    obj = Summarizer('input.txt', output)
    obj.execute()