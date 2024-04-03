import json
import numpy as np
from const import *
from nltk.tokenize import regexp_tokenize
from datasets import Dataset

def tokenize(text):
    #return word_tokenize(text, language='german')
    pattern = r'''(?x)          # set flag to allow verbose regexps
            (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
          | \w+(?:-\w+)*        # words with optional internal hyphens
          | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
          | \.\.\.              # ellipsis
          | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
        '''
    return regexp_tokenize(text, pattern)


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True)#, padding=True, truncation=True, return_tensors="pt", max_length=512)
    
    for level in ['macro_l1', 'macro_l2', 'micro_l1', 'micro_l2']:
        labels = []
        for i, label in enumerate(examples[level + '_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs[level + "_labels"] = labels
    return tokenized_inputs



def get_dataset(tokenizer, bio=True, folds=False, meta=False):
    with open('../../argschool/data/fd-lex-scriptoria/annotations/arg-school-corpus.json') as f:
        data = json.loads(f.read())
        
    # add annotations
    dataset_data = []    
    for text in data:
        obj = {}
        obj['id'] = text['id']
        obj['fold'] = text['fold']
        obj['text'] = text['text']
        obj['tokens'] = tokenize(obj['text'])
        if meta:
            obj["fdlex_id"] = text["fdlex_id"]
            obj["mzp"] = text["mzp"]

        for level in ['macro_l1', 'macro_l2', 'micro_l1', 'micro_l2']:
            label2id = get_label2id(level)

            obj[level + '_tags'] = [label2id["O"]] * len(obj['tokens'])
            annos = text[level]

            for anno in annos:
                pre = tokenize(obj['text'][:anno['start']])
                anno_text = tokenize(obj['text'][anno['start']:anno['end']])
                post = tokenize(obj['text'][anno['end']:])

                if bio:
                    obj[level + '_tags'][len(pre)] = label2id["B-" + anno['label'].replace(' ', '-')]
                    for i in range(len(pre)+1, len(pre) + len(anno_text)):
                        obj[level + '_tags'][i] = label2id["I-" + anno['label'].replace(' ', '-')]

                else:
                    obj[level + '_tags'][len(pre)] = label2id["I-" + anno['label'].replace(' ', '-')]
                    for i in range(len(pre)+1, len(pre) + len(anno_text)):
                        obj[level + '_tags'][i] = label2id["I-" + anno['label'].replace(' ', '-')]
                
                assert len(obj['tokens']) == len(obj[level + '_tags'])  
                
        for dimension in quality_dimensions:
            obj[dimension] = label2id_quality[float(text[dimension])]

        dataset_data.append(obj)
    dataset = Dataset.from_list(dataset_data)          
    
    if tokenizer:
        dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    if folds:
        fold_list = []        
        for fold in range(10):
            fold_list.append(dataset.filter(lambda x: x["fold"] == fold))
        return fold_list
            
    return dataset


'''
# 5-fold cross-validation; 1/2 fold for val/test

for fold in range(0, 10, 2):
        test_data = dataset.filter(lambda x: x["fold"] == fold)
        val_data = dataset.filter(lambda x: x["fold"] == fold+1)
        train_data = dataset.filter(lambda x: x["fold"] not in [fold, fold+1])
'''

        
    
def get_label2id(level):
    if level == 'macro_l1':
        return label2id_macro_l1
    if level == 'macro_l2':
        return label2id_macro_l2
    if level == 'micro_l1':
        return label2id_micro_l1
    if level == 'micro_l2':
        return label2id_micro_l2
    
    



# QWK, code taken from https://github.com/lingochamp/Multi-Scale-BERT-AES/blob/main/evaluate.py

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings
    
def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None): 

    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)

    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j] / num_scored_items)
            if num_ratings == 1:
                num_ratings += 0.0000001
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    if denominator <= 0.0000001:
        denominator = 0.0000001
    return 1.0 - numerator / denominator