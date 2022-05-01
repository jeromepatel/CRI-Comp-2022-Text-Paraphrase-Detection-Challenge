from audioop import reverse
import csv
from importlib.resources import path
from msilib.schema import Error
import sys
import os
from os.path import isfile,join
from typing import Set
import xml.etree.ElementTree as ET

'''
csv format
id1,id2
[doc1_id]_[para1_id]_[sen1_id],[doc1_id]_[para1_id]_[sen1_id]
'''

def read_csv(csv_name):
    csv_info = [] 
    with open(csv_name,'rt', newline='', encoding='utf-8-sig',errors='ignore') as csvfile:
        reader = csv.reader(csvfile)
        try:
            for row in reader:
                if row[0] == "id1" and row[1] == "id2": 
                    continue
                csv_info.append(row)
            return True, csv_info
        except csv.Error as e:
            print("Can't open csv file")
            return False, []    
    
def reverse(pair):
    return [pair[1],pair[0]]

def clean(ans):
    clean_ans = []
    clean_set = set()
    for each in ans:
        if not clean_set or each[0] not in clean_set:
            clean_set.add(each[0])
        else:
            continue
        if each in ans and reverse(each) in ans:
            ans.remove(reverse(each))
        clean_ans.append(each)
    return clean_ans

def get_f1score(ans,evl):
    tp,fn,fp=0,0,0;
    ans = clean(ans)
    for each in ans:
        if each in evl or reverse(each) in evl:
            tp += 1
    fp = len(ans)-tp
    fn = len(evl)-tp
    Recall = tp/(tp+fn)
    Precision=tp/(tp+fp)
    if Recall + Precision == 0:
        F1_score = 0
    else:
        F1_score = 2*(Recall*Precision) / (Recall + Precision)
    return Recall,Precision,F1_score
    
if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) >=4 and '--ans' in args and '--evl' in args:
        ans_info, evl_info = [],[]
        index_ans = args.index('--ans')
        index_evl = args.index('--evl')
        result,ans_info = read_csv(args[index_ans+1])
        if result == True:
            result,evl_info = read_csv(args[index_evl+1])
            if result == True:
                recall,precision,F1_score = get_f1score(ans_info,evl_info)
                print("Recall: " + str(recall))
                print("Precision: " + str(precision))
                print("F1-score: " + str(F1_score))
            else:
                print("Dir can't be opened")
        else:
            print("File doesn't exist")
            exit(1)
    else:
        print("File or dir path lost")
        exit(0)