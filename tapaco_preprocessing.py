import random
from datasets import load_dataset
import pandas as pd
from itertools import combinations
from src.utils import display_all
import numpy as np

from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict,concatenate_datasets


langs = ["en", "fr", "de", "es", "ar"]
seed = 420
random.seed(seed)

# Load all languages
def get_pp_pairs(ds, lang):
    """Get list of pairs of paraphrases from tapaco."""
    df = pd.DataFrame(ds)
    idx_l = df['paraphrase_set_id'].unique()
    pairs_l = []
    for i, idx in enumerate(idx_l):
        if i % 1000 == 0: print(i)
        pp_l = df.query('paraphrase_set_id==@idx')['paraphrase'].values
        # extract all combinations of pairs from the list
        pp_combinations = list(combinations(pp_l, 2))
        # randomly select 10 items from the list of tuples, or if there are less, all of them
        # unless we are doing arabic - we need all the training data we can get 
        if lang == "ar": 
            pp_pairs_sample = pp_combinations
        else: 
            pp_pairs_sample = random.sample(pp_combinations, min(len(pp_combinations), 10))
        pairs_l += pp_pairs_sample
    return pairs_l 




def balance_langs(pairs_d, k):
    """Balance the number of pairs for each language, oversampling and undersampling as needed, 
    until there are `k` pairs for each language."""
    balanced_d = {}
    for key, values in pairs_d.items():
        if len(values) > k:
            balanced_d[key] = random.sample(values, k)
        else:
            balanced_d[key] = random.choices(values, k=k)
    return balanced_d


pairs_d = dict()
for lang in langs: 
    print(lang)
    ds = load_dataset("tapaco", lang)['train']
    pairs_d[lang] = get_pp_pairs(ds,lang)
balanced_d = balance_langs(pairs_d, k=100000)
dsd_d = dict()
for lang in langs: 
    df_lang = pd.DataFrame(balanced_d[lang], columns=['source', 'target'])
    # train test split 
    df_train, df_test = train_test_split(df_lang, test_size=0.05, shuffle=True, random_state=0)
    df_valid, df_test = train_test_split(df_test, test_size=0.5, shuffle=False, random_state=0)
    dsd_d[lang] = {'train': Dataset.from_pandas(df_train.reset_index(drop=True)),
                'validation': Dataset.from_pandas(df_valid.reset_index(drop=True)),
                'test': Dataset.from_pandas(df_test.reset_index(drop=True))}
dsd = DatasetDict()  # final dsd 
for k in ['train', 'validation', 'test']:
    dsd[k] = concatenate_datasets([dsd_d[lang][k] for lang in langs])

# shuffle + tokenize 
dsd = dsd.shuffle(seed=seed)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

def tokenize_fn(batch, tokenizer):
    """Tokenize a batch of orig text using a tokenizer."""
    return tokenizer(batch["source"], text_target=batch["target"], max_length=64, padding=True, truncation=True)
    
dsd = dsd.map(tokenize_fn,  batched=True, fn_kwargs={'tokenizer': tokenizer}) 
dsd.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
# save to file 
dsd.save_to_disk("data/tapaco_processed")



