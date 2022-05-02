# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 21:30:43 2022

@author: jyotm

script for sentence transformer baseline
use python ..\evaluation.py --ans sentenceTransformer_baseline_results.csv --evl training_eval.csv
for local evaluation of training prediction
"""

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import os
import sys
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from pprint import pprint
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import sentence_transformers

# nltk.download('punkt')
N = 2000
# MODEL = 'paraphrase-albert-small-v2'
MODEL = 'all-MiniLM-L12-v2'
TRAIN = True

def read_info(file_name):
    xml_data = open(file_name,'r', encoding="utf8").read()
    root = ET.XML(xml_data)
    data = []
    for child in root:
        data.append(subchild.text for subchild in child)
    df = pd.DataFrame(data)
    df.columns = list(s.tag for s in root[0])
    return df

def create_dataset(training_files):
    dataset = {}
    #create dictionary from files
    print("loading training files.......")
    for file in tqdm(training_files):
        # print(file)
        root = 'training_text' if TRAIN else 'validation_text'
        raw_text = open(os.path.join(root,file),'r', encoding="utf8").read()
        get_sentences_with_ids(dataset,file,raw_text)
    return dataset

def get_sentences_with_ids(dataset_dct, fname,content):
    #create sent id first
    file_id = int(re.search(r"\d+", fname).group(0))
    paragraphs = content.split('\n')
    for i, para in enumerate(paragraphs):
        sentences = sent_tokenize(para)
        for j, sent in enumerate(sentences):
            if len(sent) < 20:
                continue
                # print("alert!!!!")
                # print("##"*30)
            dataset_dct[f'{file_id}_{i}_{j}'] = sent
    return dataset_dct

def create_id(file_id,para_id,sent_id):
    file_id = int(re.search(r"\d+", file_id).group(0))
    return f'{file_id}_{para_id}_{sent_id}'
    
def get_dataset_info(info_files):
    #get dataset from list of files
    dataset_info = {}
    cnt_none = 0
    print("loading training info...... ")
    for file in tqdm(training_info):
        df = read_info(os.path.join('training_info',file))
        #the df will contain file_names in columns if its empty
        if 'file_names' in df.columns:
            #skip that file in training_info
            df = None
            cnt_none += 1
        else:
            df['source_ids'] = df.apply(lambda x: create_id(x.file_ids, x.para_ids,x.sens_ids), axis=1)
            df['target_ids'] = df.apply(lambda x: create_id(x.tag_file_ids, x.tag_para_ids,x.tag_sens_ids), axis=1)
        dataset_info[file.split(".")[0]] = df
    print(f"Total no of non pleg files in dataset is: {(len(training_info)-cnt_none)} and percentage is: {(len(training_info)-cnt_none)/len(training_info)}")
    return dataset_info

def get_vector_encodings(dataset,model):
    vectors = {}
    # for key in tqdm(dataset.keys()):
    #     vectors[key] = model.encode([dataset[key]])
    #more efficient way
    vectors = model.encode(list(dataset.values()),show_progress_bar=True)
    vectors = {i:j for i,j in zip(dataset.keys(),vectors)}
    return vectors

def get_solution_parapharse_mining(model,dataset):
    #iterate over dataset to  find the cosine distances
    rev_subset = {}
    for a,b in dataset.items():
        rev_subset[b] = a
    #use inbuilt function for paraphrase mining
    solution = sentence_transformers.util.paraphrase_mining(model,list(dataset.values()),corpus_chunk_size=20000,show_progress_bar=True, top_k = 100)
    return (solution,rev_subset)
    
def get_solution_pairs(sub_dataset,dataset_info):
    pairs = {}
    source_text_ids = sub_dataset.keys()
    for k in dataset_info.keys():
        df = dataset_info[k]
        if df is not None:
            source_ids = df.source_ids.values
            target_ids = df.target_ids.values
            for x,y in zip(source_ids,target_ids):
                if x in source_text_ids:
                    pairs[x] = y
    return pairs
            
        

def get_subset_dataset(training_files,dataset_info,n=100):
    #only get subset of dataset with n files and their correspodning target files
    corresponding_target_files = {}
    subset_dataset = {}
    for file in dataset_info.keys():
        if dataset_info[file] is not None:
            corresponding_target_files[file] = dataset_info[file]['tag_file_ids'].unique()
        else:
            #the file does not contain pleg and hence the info is empty
            continue
    subset_dataset = {}
    root = 'training_text' if TRAIN else 'validation_text'
    for fname in tqdm(training_files[:n]):
        raw_text = open(os.path.join(root,fname),'r', encoding="utf8").read()
        get_sentences_with_ids(subset_dataset,fname,raw_text)
        #check if the file does contain pleg (or hence occurs in corresponding target files keys)
        key = fname.split('.')[0]
        if key in corresponding_target_files.keys():
            target_files = corresponding_target_files[key]
            for ftg in target_files:
                raw_txt_tg = open(os.path.join(root,ftg + ".newtext"),'r', encoding="utf8").read()
                get_sentences_with_ids(subset_dataset,ftg,raw_txt_tg)
    print("subset dataset created successfully.........")
    return subset_dataset

if __name__ == '__main__':
    subroot = '\\training' if TRAIN else '\\validation'
    os.chdir(r'C:\Users\jyotm\Documents\Data Science Competitions\TextParaphraseDetection\paraphrase\CRI-Comp-2022-Text-Paraphrase-Detection-Challenge\competition_data'+subroot)
    if TRAIN:
        fname = 'training_info\gen_doc0.newinfo'
        print(os.path.isfile(fname))
        df = read_info(fname)
        print(df.columns, df.shape)
        for i in df.columns:
            print(df[i])
        # print(df['tag_plag_sens'][0])
        training_files = os.listdir('training_text')
        training_info = os.listdir('training_info')
        df['source_ids'] = df.apply(lambda x: create_id(x.file_ids, x.para_ids,x.sens_ids), axis=1)
        print(df['source_ids'])

    model = SentenceTransformer(MODEL,device='cuda')
    if TRAIN:
        dataset_info = get_dataset_info(training_info)
        sub_dataset = get_subset_dataset(training_files, dataset_info, n = N)
        # sub_dataset = create_dataset(training_files)
    else:
        #for validation, we don't have info files
        val_files = os.listdir('validation_text')
        sub_dataset = create_dataset(val_files)

    # embeddings = get_vector_encodings(sub_dataset,model)
    # print(embeddings['0_1_0'])
    
    #get solution for subseet
    soln, rev_ds = get_solution_parapharse_mining(model,sub_dataset)
    
    key_ind_subset = list(sub_dataset.keys())
    soln_pairs = {}
    for (score, id1,id2) in soln:
        if score > 0.5 and key_ind_subset[id1] not in soln_pairs.keys():
            soln_pairs[key_ind_subset[id1]] = key_ind_subset[id2]
    
    save_preds = True
    if save_preds:
        val_df = pd.DataFrame(soln_pairs.items())
        val_df.columns = ['id1','id2']
        val_df.to_csv("..\sentenceTransformer_baseline_results.csv",index=False)
    
    if TRAIN:
        gt_pairs = get_solution_pairs(sub_dataset, dataset_info)
        cnt = 0
        total = len(gt_pairs)
        for sol in soln_pairs.keys():
            if sol in gt_pairs.keys():
                if soln_pairs[sol] == gt_pairs[sol]:
                    cnt += 1
        
        print(f"The accuracy of this method on subset is: {cnt/total}")
    
    # print(dataset)
    print("#"*20)
    #verify the dataset creation by keys from info file
    # ids = ['0_'+str(i)+'_'+str(j) for i,j in zip(list(df.para_ids.values),list(df.sens_ids.values))]
    # print(ids)
    # print([dataset[sen] for sen in ids])
    # print(df.plag_sens.values)
    # print("#"*20)
    # print([dataset[s] for s in dataset.keys() if s[2:3] == '0'])
    # print(dataset.keys())
    # print(dataset['0_1_0'])
    
    
    
'''

Results:
    on training set: (The scores are calculated using evaluation.py script against training_eval.csv )
        subset size | Model | recall | f1 score | precision | other hyper parameter imp change (time to calculate batches embeddings)
        1000 | 'all-MiniLM-L12-v2' | 0.02 | 0.01 | 0.01 | with corpus_chunk_size = 1k and top_k = 2 (something went wrong here, dk)
        
        1000 | 'all-MiniLM-L12-v2' | 0.289 | 0.058 | 0.032 | with corpus_chunk_size = 10k and top_k = 2  (time: 3:11)
        1000 | 'all-MiniLM-L12-v2' | 0.289 | 0.061 | 0.035 | with corpus_chunk_size = 10k and top_k = 50 (Time: 2:53)
        1000 | 'all-MiniLM-L12-v2' | 0.289 | 0.062 | 0.034 | with corpus_chunk_size = 20k and top_k = 50 (Time: 3:00)
        2000 | 'all-MiniLM-L12-v2' | 0.502 | 0.050 | 0.091 | with corpus_chunk_size = 20k and top_k = 100 (Time: 5:06)
        full dataset | 'all-MiniLM-L12-v2' | 0.653 | 0.059 | 0.109 | with corpus_chunk_size = 20k and top_k = 100 (Time: 6:45 ) + misc time: 4:30
        
        1000 | 'paraphrase-albert-small-v2' | 0.29 | 0.067 | 0.04 | with corpus_chunk_size = 10k and top_k = 50 (Time: 6:25)
        
        
'''