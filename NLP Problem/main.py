
import nltk
#import pandas as pd

import csv
import re

import os

import sys
from nltk.tokenize import word_tokenize

nltk.download('averaged_perceptron_tagger')

data_path = 'SubtaskA_EvaluationData.csv'
out_path = 'Anshuman_Pati.csv'
#print('Testing')

class taggingParsing:

    def sentenceSplit(self, text):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(text)
        return sentences

    def taggingNLTK(self, text):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(text)
        for sent in sentences:
            text = word_tokenize(sent)
            tagged_sent = nltk.pos_tag(text)


def classify(sent_list):

    # Define keywords and regular expressions which indicate a suggestion i.e. X=1. If none of these match, X=0.
    
    keywords = ["suggest","recommend","hopefully","go for","request","it would be nice", "It would be nice", "adding", "Adding", "Should","could come with", "could be", "I need" , "we need", "We need", "needs", "would like to","would love to","allow", "Allow" ,"add", "Add", "don't", "please", "Please", "alternative", "should", "must"]

    pattern_strings = [r'.*would\slike.*if.*', r'.*i\swish.*', r'.*i\shope.*', r'.*I\swish.*', r'.*I\shope.*', r'.*i\swant.*', r'.*hopefully.*',
                       r".*if\sonly.*", r".*would\sbe\sbetter\sif.*", r".*should.*", r".*would\sthat.*",
                       r".*can't\sbelieve.*didn't.*", r".*don't\sbelieve.*didn't.*", r".*do\swant.*", r".*i\scan\shas.*"]


    compiled_patterns = []      
    for patt in pattern_strings:
        compiled_patterns.append(re.compile(patt))

    # Store the labelled patterns
    # Default label set to zero
    label_list = []
    for sent in sent_list:
        tokenized_sent = word_tokenize(sent[1])
        tagged_sent = nltk.pos_tag(tokenized_sent)
        tags = [i[1] for i in tagged_sent]
        label = 0
        patt_matched = False
        for compiled_patt in compiled_patterns:
            joined_sent = " ".join(tokenized_sent)
            matches = compiled_patt.findall(joined_sent)
            if len(matches) > 0:
                patt_matched = True
        keyword_match = any(elem in keywords for elem in tokenized_sent)
        
        
        pos_match = any(elem in ['MD', 'VB'] for elem in tags)

        
        # Label set to 1 (indicates 'suggestion') if the given review matches the string or regex expressions
        if patt_matched:
            label = 1
        elif keyword_match == True:
                label = 1
        elif pos_match == True:
                label = 1    
     

        label_list.append(label)



    return label_list

#This reads CSV a given CSV and stores the data in a list
def read_csv(data_path):
    file_reader = csv.reader(open(data_path,"rt"), delimiter=',') #, errors="ignore", encoding="utf-8"
    sent_list = []

    for row in file_reader:
        id = row[0]
        sent = row[1]
        sent_list.append((id,sent))
    return sent_list

#This will create and write into a new CSV
def write_csv(sent_list, label_list, out_path):
        filewriter = csv.writer(open(out_path, "w+"))
        count = 0
        for ((id, sent), label) in zip(sent_list, label_list):
                filewriter.writerow([id, sent, label])


if __name__ == '__main__':
    sent_list = read_csv(data_path)
    label_list = classify(sent_list)
    write_csv(sent_list, label_list, out_path)

