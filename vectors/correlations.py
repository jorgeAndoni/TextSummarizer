import os
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from biokit.viz import corrplot
import string
import numpy as np 
import pandas as pd
import statsmodels.api as sm
from grafico import plot_corr_ellipses 

def to_int(vector):
	vec = []
	for i in vector:
		vec.append(int(i))
	return vec

def read_file(file):
	key = file
	key = key[key.find('/')+1:] 
	key = key[:key.find('.')]
	arc = open(file, 'r')
	vectors = []
	for i in arc:
		i = i.rstrip('\n')
		auxi = i
		i = i[1:len(i)-1] 
		i = i.split(',') 
		vector =  to_int(i)
		vectors.append(vector)

	return (key, vectors) 

def get_corpus_data(location):
	dictionary = dict()
	files = os.listdir(location)

	for i in files:
		data = read_file(location + i)
		key = data[0]
		value = data[1]
		dictionary[key] = value

	return dictionary


def get_correlations(vector1, vector2):
	avg = 0.0
	for i in range(len(vector1)):
		avg+= spearmanr(vector1[i] , vector2[i])[0]  
	return round(avg/float(len(vector1)),2)   


def generate_correlation(data):
	measures = ['dg', 'stg', 'sp', 'sp_w', 'pr', 'pr_w', 'accs', 'gaccs', 'sym', 'at']
	names = ['dg', 'stg', 'sp', 'sp_w', 'pr', 'pr_w', 'access', 'gAccess', 'sym', 'absT']
	matrix_correlations = []

	for index, measure in enumerate(measures):
		print index, measure
	
	for measure in measures:
		temporal = data[measure]
		vector = []
		for temp_measure in measures:
			temp_vector = data[temp_measure]
			correlation = get_correlations(temporal, temp_vector)
			vector.append(correlation) 
		matrix_correlations.append(vector)


	for i in matrix_correlations:
		print i 
	#letters = string.uppercase[0:10]
	#print dict( ( (k, np.random.random(10)+ord(k)-65) for k in letters))

	dictionary = dict()
	for index, measure in enumerate(measures):
		dictionary[measure] = matrix_correlations[index]

	df = pd.DataFrame(matrix_correlations) 
	#df = df.corr()
	
	#fig, ax = plt.subplots(1, 1)
	#m = plot_corr_ellipses(df, ax=ax, cmap='seismic')
	#cb = fig.colorbar(m)
	#cb.set_label('Correlation coefficient')
	#ax.margins(0.1)

	

	c = corrplot.Corrplot(df)
	
	c.plot(lower='ellipse', cmap='hsv' ) # hsv gist_rainbow jet

	#value = np.asarray(matrix_correlations)

	t1 = 'Matrix of Spearman correlation for CSTNews'
	t2 = 'Matrix of Spearman correlation for DUC-2002'
	t3 = 'Matrix of Spearman correlation for DUC-2004'

	
	#sm.graphics.plot_corr(value, xnames=names, title=t3)


	
	plt.show()
		
		





if __name__ == '__main__':
	
	cst_files = 'cst_news/'
	duc2002_files = 'duc2002/'
	duc2004_files = 'duc2004/'



	data = get_corpus_data(duc2004_files)
	
	generate_correlation(data)



	