import os
import evaluate
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import set_seed, AutoTokenizer, DebertaForTokenClassification, TrainingArguments, Trainer, AdapterTrainer, DataCollatorForTokenClassification, DebertaV2AdapterModel

from const import *
from utils import *

set_seed(758)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seqeval = evaluate.load("seqeval")
model_path = '../../transformer-models/mdeberta-v3-base' # microsoft/mdeberta-v3-base
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

learning_rate = 1e-4
batch_size = 16
warmup_steps = 500


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[list(label2id.keys())[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[list(label2id.keys())[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return results  


def experiment(level):
    print(level)
    if os.path.isfile(f'hyperparam-tuning-results/structure-mdeberta-adapter-{level}.csv'):
        results_list = list(pd.read_csv(f'hyperparam-tuning-results/structure-mdeberta-adapter-{level}.csv', index_col=0)[['fold']].itertuples(index=False, name=None))
        results = pd.read_csv(f'hyperparam-tuning-results/structure-mdeberta-adapter-{level}.csv', index_col=0).to_dict('records')
    else:
        results_list = []
        results = []
        
    for fold in range(0, 10, 2):
        print("Starting fold", fold)   
        
        if not fold in results_list: # check if done already
            test_data = dataset.filter(lambda x: x["fold"] == fold)
            val_data = dataset.filter(lambda x: x["fold"] == fold+1)
            train_data = dataset.filter(lambda x: x["fold"] not in [fold, fold+1])              

            # model
            model = DebertaV2AdapterModel.from_pretrained( # DebertaForTokenClassification
                model_path, 
                num_labels = len(id2label), 
                id2label = id2label,
                label2id = label2id,
                ignore_mismatched_sizes=True
            )
            # add new adapter
            model.add_adapter(f"{level}_adapter")
            # add matching tagging head
            model.add_tagging_head(
                f"{level}_adapter",
                num_labels = len(id2label), 
                id2label = id2label,
            )
            # activate adapter
            model.train_adapter(f"{level}_adapter")
            model.to(device)                        

            # hyperparameter
            training_args = TrainingArguments(
                output_dir=f"../models/mdeberta-adapter-{level}",
                overwrite_output_dir=True,
                num_train_epochs=50,
                weight_decay=0.01,
                adam_epsilon=1e-6,
                adam_beta1=0.9,
                adam_beta2=0.999,
                max_grad_norm=1.0,
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=warmup_steps,
                evaluation_strategy="steps",
                save_strategy="steps",   
                logging_strategy="steps",  
                load_best_model_at_end=True,
                metric_for_best_model='eval_overall_f1',
                greater_is_better=True
            )

            trainer = AdapterTrainer(
                model=model,                     
                args=training_args,              
                train_dataset=train_data,    
                eval_dataset=val_data,        
                tokenizer=tokenizer,  
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            trainer.train()                    

            # eval
            preds = trainer.predict(test_data)

            cur_result = preds.metrics
            cur_result['fold'] = fold
            results.append(cur_result)

            results_df = pd.json_normalize(results)
            results_df.to_csv(f'hyperparam-tuning-results/structure-mdeberta-adapter-{level}.csv')
        
    results_df = pd.json_normalize(results)
    results_df.to_csv(f'hyperparam-tuning-results/structure-mdeberta-adapter-{level}.csv')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', '-l', default='macro_l1')
    args = parser.parse_args()
    
    dataset = get_dataset(tokenizer)
    if args.level == 'macro_l1':
        id2label = id2label_macro_l1
        label2id = label2id_macro_l1
        dataset = dataset.add_column("labels", dataset['macro_l1_labels'])
    elif args.level == 'macro_l2':
        id2label = id2label_macro_l2
        label2id = label2id_macro_l2
        dataset = dataset.add_column("labels", dataset['macro_l2_labels'])
    elif args.level == 'micro_l1':
        id2label = id2label_micro_l1
        label2id = label2id_micro_l1
        dataset = dataset.add_column("labels", dataset['micro_l1_labels'])
    elif args.level == 'micro_l2':
        id2label = id2label_micro_l2
        label2id = label2id_micro_l2
        dataset = dataset.add_column("labels", dataset['micro_l2_labels'])
    else:
        print(f'error: level {args.level} does not exist')

    experiment(args.level)

