import os
import evaluate
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import set_seed, AutoTokenizer, DebertaForSequenceClassification, TrainingArguments, Trainer, AdapterTrainer, DebertaV2AdapterModel
from transformers.adapters import Fuse

from const import *
from utils import *

set_seed(758)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = '../../transformer-models/mdeberta-v3-base' # microsoft/mdeberta-v3-base
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

learning_rate = 5e-5 #1e-4
batch_size = 16
warmup_steps = 500
    
    
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    
    report = classification_report(labels, predictions, output_dict=True)
    macro_score = report.pop("macro avg")
    report['macro-f1-score'] = macro_score["f1-score"]
    report['qwk'] = quadratic_weighted_kappa(labels, predictions)
    return report 


def experiment(level, setting):
    print(level, setting)
    if os.path.isfile(f'hyperparam-tuning-results/quality-fusion-{level}-{setting}.csv'):
        results_list = list(pd.read_csv(f'hyperparam-tuning-results/quality-fusion-{level}-{setting}.csv', index_col=0)[['fold']].itertuples(index=False, name=None))
        results = pd.read_csv(f'hyperparam-tuning-results/quality-fusion-{level}-{setting}.csv', index_col=0).to_dict('records')
    else:
        results_list = []
        results = []        

    for fold in range(0, 10, 2):
        print("Starting fold", fold)
        
        if not fold in results_list: # check if done already
            test_data = dataset.filter(lambda x: x["fold"] == fold)
            val_data = dataset.filter(lambda x: x["fold"] == fold+1)
            train_data = dataset.filter(lambda x: x["fold"] not in [fold, fold+1])  

            # train
            model = DebertaV2AdapterModel.from_pretrained(
                model_path, 
                num_labels = len(id2label_quality), 
                id2label = id2label_quality,
                label2id = label2id_quality,
                ignore_mismatched_sizes=True
            )
            
            # load structure adapters
            for structure_level in ['macro_l1', 'macro_l2', 'micro_l1', 'micro_l2']:
                model.load_adapter(f"../models/mdeberta-adapter-{structure_level}-final", with_head=False)
                
            # add classification head
            model.add_classification_head(
                f"{level}",
                num_labels = len(id2label_quality), 
                id2label = id2label_quality,
            )
            
            # adapter fusion
            if setting == 'all':
                adapter_setup = Fuse('macro_l1_adapter', 'macro_l2_adapter', 'micro_l1_adapter', 'micro_l2_adapter')
            elif setting == 'macro':
                adapter_setup = Fuse('macro_l1_adapter', 'macro_l2_adapter')
            elif setting == 'micro':
                adapter_setup = Fuse('micro_l1_adapter', 'micro_l2_adapter')
            elif setting == 'macro_l1':
                adapter_setup = Fuse('macro_l1_adapter')
            elif setting == 'macro_l2':
                adapter_setup = Fuse('macro_l2_adapter')
            elif setting == 'micro_l1':
                adapter_setup = Fuse('micro_l1_adapter')
            elif setting == 'micro_l2':
                adapter_setup = Fuse('micro_l2_adapter')
            else:
                raise ValueError(f'{setting} unknown!')
                
            model.add_adapter_fusion(adapter_setup, "dynamic", {"regularization": True})
            model.set_active_adapters(adapter_setup)            
            model.train_fusion(adapter_setup)
            model.to(device)

            # train
            training_args = TrainingArguments(
                output_dir=f"../models/mdeberta-fusion-{level}-{setting}",
                overwrite_output_dir=True,
                num_train_epochs=20, #50,
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
                metric_for_best_model='eval_qwk',
                greater_is_better=True
            )

            trainer = AdapterTrainer(
                model=model,                     
                args=training_args,              
                train_dataset=train_data,    
                eval_dataset=val_data,        
                tokenizer=tokenizer,  
                compute_metrics=compute_metrics,
            )
            trainer.train()                    

            # eval
            preds = trainer.predict(test_data)

            cur_result = preds.metrics
            cur_result['fold'] = fold
            results.append(cur_result)

            results_df = pd.json_normalize(results)
            results_df.to_csv(f'hyperparam-tuning-results/quality-fusion-{level}-{setting}.csv')                        
        
    results_df = pd.json_normalize(results)
    results_df.to_csv(f'hyperparam-tuning-results/quality-fusion-{level}-{setting}.csv')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', '-l', default='gesamteindruck')
    parser.add_argument('--setting', '-s', default='all') # as before
    args = parser.parse_args()
    
    dataset = get_dataset(tokenizer)
    dataset = dataset.add_column("labels", dataset[args.level])

    experiment(args.level, args.setting)

