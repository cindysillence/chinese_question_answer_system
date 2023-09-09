import pandas as pd
import gensim
import jieba
import numpy as np
import argparse
import os
import joblib
from text_classification.predict_textcnn import classification_predict
from text_similarity.predict import predict
from chitchat.interact import chitchat
df = pd.read_csv('data/insurance_qa.csv')

question = df['question'].values
answer = df['answer'].values


model = gensim.models.Word2Vec.load('data/word2vec/wiki.model')




# transfer a sentence to vector
def sen2vec(text):

    vec = np.zeros(100)
    segment = list(jieba.cut(text))
    for s in segment:
        try:
            vec += model.wv[s]
        except:
            # I used wiki dataset to train the word vector, but some specific word could not be exist in the wiki dataset
            # will be better to train the specific word with wiki together
            pass
    vec = vec / len(segment)
    return vec

def cosine(a, b):
    if not isinstance(b,np.ndarray):
        b = np.array(b)
    return np.matmul(a, np.array(b).T) / np.linalg.norm(a) / np.linalg.norm(b, axis=1)


question_vec = []
for q in question:
    question_vec.append(sen2vec(q))


def qa(text):
    out = classification_predict(text)[0]
    if out > 0.5:
        print('this is a chatbot')
        print(chitchat(text))
        return chitchat(text)
        #continue
    else:
        vec = sen2vec(text)

        # compute similarity
        similarity = cosine(vec, question_vec)
        #
        max_similarity = max(similarity)
        print('similarity', similarity)
        if max_similarity < 0.8:
            print('no answer has been found')
            return 'no answer has been found'
        else:
            # return top 10 index
            top_10 = np.argsort(-similarity)[0:10]
            # check all 10 candidates
            candidate = question[top_10]
            # use text similarity
            esim_res=predict([text]*10,candidate)
            index_dic = {}

            print('candidate result')
            for i, index in enumerate(top_10):
                print(candidate[i],'cosin similarity',similarity[index],'text similarity',esim_res[i])
                index_dic[i] = index

            esim_index = np.argmax(esim_res)
            print('the most same question:', question[index_dic[esim_index]])
            print('answer:',answer[index_dic[esim_index]] )
            return answer[index_dic[esim_index]]




if __name__ == '__main__':
    while 1:
        q = input('please give your question：')
        qa(q)
