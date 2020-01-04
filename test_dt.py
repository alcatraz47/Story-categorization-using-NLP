import os
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import multiprocessing
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import scikitplot as skplt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from scipy import spatial
from bs4 import BeautifulSoup
from sklearn.externals.six import StringIO  
from IPython.display import Image, display  
from sklearn.tree import export_graphviz
import pydotplus
import warnings
warnings.filterwarnings('ignore')
cores = multiprocessing.cpu_count()
os.listdir()


# df = pd.read_csv(os.getcwd() + "/Collective_Dataset/419_data - Sheet1.csv", usecols=[0,1])#header=None, , names=['story', 'category']

# df = df.sample(frac=1).reset_index(drop=True)

# df.index = range(270)

# df.story.apply(lambda x: len(x.split(' '))).sum()

# def cleanText(text):
#     text = BeautifulSoup(text, "lxml").text
#     text = re.sub(r'\|\|\|', r' ', text) 
#     text = re.sub(r'http\S+', r'<URL>', text)
#     text = text.lower()
#     text = text.replace('x', '')
#     return text
# df['story'] = df['story'].apply(cleanText)

# def tokenize_text(text):
#     tokens = []
#     for sent in nltk.sent_tokenize(text):
#         for word in nltk.word_tokenize(sent):
#             if len(word) < 2:
#                 continue
#             tokens.append(word.lower())
#     return tokens

# X = df['story']
# y = df['category']
# ten_fold = KFold(n_splits=5, shuffle = True, random_state=42)
# total_fold = ten_fold.get_n_splits(X)

# fold_no = 1
# print("Total Fold No.: {}\n\n" .format(total_fold))
# for train_index, test_index in ten_fold.split(X):
# #     print("Train Fold No.: ", train_index, " Test Fold No.: ", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
    
#     print("No. of Training Dataset in Fold ", fold_no, ": ", len(X_train))
#     print("No. of Testing Dataset in Fold ", fold_no, ": ", len(X_test))
    
#     X_train = list(zip(X_train,y_train))
#     X_train = pd.DataFrame(X_train, columns=['story', 'category'])
    
#     X_test = list(zip(X_test,y_test))
#     X_test = pd.DataFrame(X_test, columns=['story', 'category'])
    
#     train_tagged = X_train.apply(lambda r: TaggedDocument(words=tokenize_text(r['story']), tags=[r['category']]), axis=1)
#     test_tagged = X_test.apply(lambda r: TaggedDocument(words=tokenize_text(r['story']), tags=[r['category']]), axis=1)
    
#     #PV_DBOW using DT with Entropy
#     model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
#     model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])
    
#     for epoch in range(5):
#         model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
#         model_dbow.alpha -= 0.002
#         model_dbow.min_alpha = model_dbow.alpha
    
#     def vec_for_learning(model, tagged_docs):
#         sents = tagged_docs.values
#         targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
#         return targets, regressors
    
#     y_trained, X_trained = vec_for_learning(model_dbow, train_tagged)
#     y_tested, X_tested = vec_for_learning(model_dbow, test_tagged)
    
#     dt_dbow_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 10, max_depth = 3, min_samples_leaf = 5)
#     dt_dbow_entropy.fit(X_trained, y_trained)
#     y_pred = dt_dbow_entropy.predict(X_tested)

#     # print("FOR PV_DBOW Using Decision Tree with Entropy: ")
#     # print("Fold No.: ", fold_no)
#     # print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
#     # print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
#     # print('\nTesting Confusion Matrix: ')
#     # print(confusion_matrix(y_test, y_pred),"\n")
#     # print('Testing Classification Report: ')
#     # print(classification_report(y_test, y_pred))    
#     # print("\n\n")

#     dot_data = StringIO()
#     export_graphviz(dt_dbow_entropy, out_file=dot_data, filled=True, rounded=True, special_characters=True)
#     # print(dot_data)
#     graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#     # Image(graph.create_png())
#     display(graph)
    
#     # #PV_DM
#     # model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
#     # model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])
    
#     # for epoch in range(5):
#     #     model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
#     #     model_dmm.alpha -= 0.002
#     #     model_dmm.min_alpha = model_dmm.alpha
    
#     # y_trained, X_trained = vec_for_learning(model_dmm, train_tagged)
#     # y_tested, X_tested = vec_for_learning(model_dmm, test_tagged)
    
#     # dt_dbow_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 10, max_depth = 3, min_samples_leaf = 5)
#     # dt_dbow_entropy.fit(X_trained, y_trained)
#     # y_pred = dt_dbow_entropy.predict(X_tested)
    
#     # print("FOR PV_DM Using Decision Tree with Entropy:  ")
#     # print("Fold No.: ", fold_no)
#     # print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
#     # print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
#     # print('\nTesting Confusion Matrix: ')
#     # print(confusion_matrix(y_test, y_pred),"\n")
#     # print('Testing Classification Report: ')
#     # print(classification_report(y_test, y_pred))    
#     # print("\n\n")
    
    
#     # #FOR PAIRED_MODEL
#     # model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
#     # model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    
#     # new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])
    
#     # y_train, X_train = vec_for_learning(new_model, train_tagged)
#     # y_test, X_test = vec_for_learning(new_model, test_tagged)
    
#     # dt_dbow_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 10, max_depth = 3, min_samples_leaf = 5)
#     # dt_dbow_entropy.fit(X_trained, y_trained)
#     # y_pred = dt_dbow_entropy.predict(X_tested)
    
#     # print("FOR Paired Model Using Decision Tree with Entropy: ")
#     # print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
#     # print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))
#     # print('\nTesting Confusion Matrix: ')
#     # print(confusion_matrix(y_test, y_pred),"\n")
#     # print('Testing Classification Report: ')
#     # print(classification_report(y_test, y_pred))
#     fold_no+=1
extra_pred = [0, 2, 1, 1, 1, 0, 0, 0, 1]
extra_category = [0, 2, 2, 1, 2, 0, 0, 1, 1]
plt.rcdefaults()
# matrix = confusion_matrix(extra_category, extra_pred)
# sns.heatmap(matrix,annot=True,cbar=True)
# fix for mpl bug that cuts off top/bottom of seaborn viz
# b, t = plt.ylim() # discover the values for bottom and top
# b += 0.5 # Add 0.5 to the bottom
# t -= 0.5 # Subtract 0.5 from the top
# plt.ylim(b, t) # update the ylim(bottom, top) values

# plt.ylabel('True Label')
# plt.xlabel('Predicted Label')
# plt.title('Confusion Matrix')
# temp = "Decision Tree With PV-DBOW Using Gini Confusion Matrix Heatmap On Test Set"+'.png'
# plt.show()
# plt.savefig(temp, pad_inches=0.1)

objects = ('Gaussian NB', 'DT-Gini-PV-DBOW', 'DT-Entropy-PV-DBOW', 'KNN')
y_pos = np.arange(len(objects))
performance = [88.80, 94.00, 94.50, 94.54]

plt.barh(y_pos, performance, align='center', alpha=0.5)
plt.yticks(y_pos, objects)
plt.xlabel('F1 Score')
plt.title('Model Bar Plot According To F1 Score')

plt.show()