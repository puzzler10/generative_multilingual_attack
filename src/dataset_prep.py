from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, Subset, DataLoader
from torch.cuda import empty_cache
from torch import tensor, gather
from random import Random
from datasets import load_dataset, DatasetDict
from datasets import disable_caching
import logging
import gc 
import src.models 
from src.utils import *
import copy

logger = logging.getLogger(__name__)

DS_INFO = {
    'tweet_sentiment_multilingual': {
        'task': 'sentiment',
        'LABEL2ID': {'negative': 0, 'neutral': 1, 'positive': 2},
        'ID2LABEL': {0: 'negative', 1: 'neutral', 2: 'positive'},
        'text_field': 'text',
        'label_field': 'label',
        'num_labels': 3
    },
    'amazon_reviews_multi': {  # stats are after pos/neg processing
        'task': 'sentiment',
        'LABEL2ID': {'0': 0, '1': 1},
        'ID2LABEL': {0: '0', 1: '1'},
        'text_field': 'text',
        'label_field': 'label',
        'num_labels': 2
    }, 
    # DUMMY ONE, JUST FOR MT5 pretraining
    'combined': {
        'task': 'sentiment',
                'LABEL2ID': {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4},
                'ID2LABEL': {0: '0', 1: '1', 2: '2', 3: '3', 4: '4'},
                'text_field': 'text',
                'label_field': 'label',
                'num_labels': 5
    }
}
LANG_MAP = {
    'ar': 'arabic',
    'en': 'english', 
    'fr': 'french',
    'es': 'spanish',
    'de': 'german'
}
LANGS = ['en', 'de', 'es', 'fr', 'ar']

class BaseDataset(Dataset):  
    """Common functions for both AdversaryDataset and VictimDataset"""
    def load_data(self): 
        if self.ds_name == "tweet_sentiment_multilingual":
            if self.args.lang == "all": return self._load_tweet_sentiment_multilingual_all()
            else:                       return self._load_tweet_sentiment_multilingual(self.args.lang)
        elif self.ds_name == "amazon_reviews_multi":
            if self.args.lang == "all": return self._load_amazon_reviews_multi_all()
            else:                       return self._load_amazon_reviews_multi(self.args.lang)

    def _load_tweet_sentiment_multilingual(self, lang): 
        """language_expanded: english, not en; french, not fr"""
        dsd = load_dataset("data/tweet_sentiment_multilingual/tweet_sentiment_multilingual.py", LANG_MAP[lang])
        dsd['train'] = dsd['train'].add_column('lang', [lang]*len(dsd['train']))
        dsd['validation'] = dsd['validation'].add_column('lang', [lang]*len(dsd['validation']))
        dsd['test'] = dsd['test'].add_column('lang', [lang]*len(dsd['test']))
        return dsd
    
    def _load_tweet_sentiment_multilingual_all(self): 
        train_l, val_l, test_l = [], [], []
        for lang in LANGS:
            dsd = self._load_tweet_sentiment_multilingual(lang)
            train_l.append(dsd['train'])
            val_l.append(dsd['validation'])
            test_l.append(dsd['test'])
        dsd = DatasetDict({'train': concatenate_datasets(train_l), 'validation': concatenate_datasets(val_l), 'test': concatenate_datasets(test_l)})
        return dsd 
    
    def _load_amazon_reviews_multi(self, lang): 
        ds_path = f"data/amazon_reviews_multi/{lang}"
        ds_train = load_dataset("json", data_files=f"{ds_path}/train.jsonl")
        ds_validation = load_dataset("json", data_files=f"{ds_path}/validation.jsonl")
        ds_test = load_dataset("json", data_files=f"{ds_path}/test.jsonl")
        dsd = DatasetDict({'train': ds_train['train'], 'validation': ds_validation['train'], 'test': ds_test['train']})
        dsd = dsd.remove_columns(["label_text"])
        dsd = dsd.remove_columns(["id"])
        # remove label 3 
        dsd = dsd.filter(lambda example: example['label'] != 3)
        # Map 1 and 2 to negative, 4 and 5 to positive, remove 3
        dsd = dsd.map(lambda example: {'label': 0 if example['label'] <= 2 else 1}, remove_columns=['label'])
        dsd['train'] = dsd['train'].add_column('lang', [lang]*len(dsd['train']))
        dsd['validation'] = dsd['validation'].add_column('lang', [lang]*len(dsd['validation']))
        dsd['test'] = dsd['test'].add_column('lang', [lang]*len(dsd['test']))
        return dsd

    def _load_amazon_reviews_multi_all(self): 
        train_l, val_l, test_l = [], [], []
        for lang in ['de','en','es','fr']:
            dsd = self._load_amazon_reviews_multi(lang)
            train_l.append(dsd['train'])
            val_l.append(dsd['validation'])
            test_l.append(dsd['test'])
        dsd = DatasetDict({'train': concatenate_datasets(train_l), 'validation': concatenate_datasets(val_l), 'test': concatenate_datasets(test_l)})
        return dsd

    def add_idx(self, batch, idx):
        """Add row numbers"""
        batch['idx'] = idx
        return batch
    
    def add_n_tokens(self, batch, field):
        """Add the number of tokens present in the tokenised text """
        batch['n_tokens'] = [len(o) for o in batch[field]]
        return batch

    def get_dataloaders_dict(self, dsd, args, collate_fn):
        """Prepare a dict of dataloaders for train, valid and test"""
        if args.bucket_by_length and args.shuffle_train:  raise Exception("Can only do one of bucket by length or shuffle")
        persistent_workers = True if args.num_workers > 0 else False
        d = dict()
        for split, ds in dsd.items():
            batch_size = args.batch_size if split == "train" else args.batch_size_eval
            drop_last = True if len(ds) % batch_size == 1 else False
            if args.shuffle_train:
                if split == "train":
                    d[split] =  DataLoader(ds, batch_size=batch_size,
                                           shuffle=True,
                                            collate_fn=collate_fn, drop_last=drop_last,
                                           num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent_workers)
                else:
                    d[split] =  DataLoader(ds, batch_size=batch_size,
                                               shuffle=False, collate_fn=collate_fn, drop_last=drop_last,
                                           num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent_workers)                                       

            if args.bucket_by_length:
                if args.shuffle_buckets: 
                    # Sort the dataset by token count,  group sorted indices into batches, shuffle, flatten, make new ds
                    sorted_indices = sorted(range(len(ds)), key=lambda i: ds[i]['n_tokens'])
                    batches = [sorted_indices[i:i+batch_size] for i in range(0, len(sorted_indices), batch_size)]
                    Random(x=args.seed).shuffle(batches)  # x = seed for random
                    shuffled_indices = [i for batch in batches for i in batch]
                    shuffled_ds = Subset(ds, shuffled_indices)
                    d[split] = DataLoader(shuffled_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=drop_last,
                                          num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent_workers)
                else: 
                    d[split] =  DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, drop_last=drop_last,
                                       num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent_workers)
        return d

    def get_train_valid_test_split(self, dsd, train_size=0.8):
        dsd1 = dsd['train'].train_test_split(train_size=train_size, shuffle=True, seed=0)
        dsd2 = dsd1['test'].train_test_split(train_size=0.5, shuffle=True, seed=0)
        return DatasetDict({
            'train': dsd1['train'],
            'validation': dsd2['train'],
            'test': dsd2['test']
        })
    
    def get_train_valid_split(self, dsd, train_size=0.8): 
        dsd1 = dsd['train'].train_test_split(train_size=train_size, shuffle=True, seed=0)
        return DatasetDict({
            'train': dsd1['train'],
            'validation': dsd1['test'],
            'test': dsd['test']
        })
    
    def update_dataset_info(self, dsd, dld=None): 
        """Make a dict with dataset stats. Used later for wandb logging. Useful for debugging """
        d = dict()
        for ds_name in ['train', 'validation', 'test']:
            d[f"ds_{ds_name}_len"] = len(dsd[ds_name])
            for i in range(self.num_labels):
                d[f"ds_{ds_name}_class_{self.ID2LABEL[i]}"] =  len(dsd[ds_name].filter(lambda x: x['label'] == i))
                if dld is not None: d[f"ds_{ds_name}_num_batches"] = len(dld[ds_name])
        d['ds_num_labels']    = self.num_labels
        d['ds_text_field']    = self.text_field
        d['ds_label_field']   = self.label_field
        self.dataset_info = d

class AdversaryDataset(BaseDataset):
    """Class with methods for the multilingual whitebox adversary""" 
    def __init__(self, args, pp_tokenizer, vm_tokenizer, ld_tokenizer, vm_model, sts_model, ld_model,  get_dataloaders=True): 
        self.ds_name = args.dataset_name
        for k, v in DS_INFO[self.ds_name].items(): setattr(self, k, v)
        self.args = args
        self.pp_tokenizer = pp_tokenizer
        self.vm_tokenizer = vm_tokenizer
        self.ld_tokenizer = ld_tokenizer

        self.vm_model = vm_model
        self.sts_model = sts_model
        self.ld_model = ld_model
        if     args.accelerator == "gpu": self.device = 'cuda' 
        elif   args.accelerator == "cpu": self.device = 'cpu'
        else: raise Exception('only "cpu" and "gpu" supported for --accelerator argument.') 
        self.vm_model   = self.vm_model.to(self.device) 
        self.sts_model  = self.sts_model.to(self.device) 
        self.ld_model  = self.ld_model.to(self.device) 
        if self.args.disable_hf_caching: disable_caching()
        dsd = self.load_data()
        self.dsd,self.dld = self.prepare_data(dsd, get_dataloaders=get_dataloaders)
        del self.vm_model
        del self.sts_model
        del self.ld_model
        empty_cache()
        gc.collect()

    def prep_input_for_t5_paraphraser(self, batch, task):
        """To paraphrase the t5 model needs a "paraphrase: " prefix. 
        See the appendix of the T5 paper for the prefixes. (https://arxiv.org/abs/1910.10683) """  
        if  task == 'paraphrase':  
            batch[f'{self.text_field}_with_prefix'] = [src.models.T5_PREFIX[task]  + sen for sen in batch[self.text_field]]
        else:  
            raise Exception("shouldn't get here")                                      
        return batch

    def add_sts_embeddings(self, batch):
        """Calculate and save sts embeddings of the original text"""
        batch['orig_sts_embeddings'] = self.sts_model.encode(batch[self.text_field], batch_size=64, convert_to_tensor=False)
        return batch

    def tokenize_fn(self, batch, tokenizer, use_prefix):
        """Tokenize a batch of orig text using a tokenizer."""
        text_field = f'{self.text_field}_with_prefix' if use_prefix else self.text_field
        if self.args.long_example_behaviour == 'remove':    
            return tokenizer(batch[text_field])  # we drop the long examples later
        elif self.args.long_example_behaviour == 'truncate':         
            return tokenizer(batch[text_field], truncation=True, max_length=self.args.max_length_orig)

    def collate_fn(self, x):
        """Collate function used by the DataLoader that serves tokenized data.
        x is a list (with length batch_size) of dicts. Keys should be the same across dicts.
        I guess an error is raised if not. """
        # check all keys are the same in the list. the assert is quick (~1e-5 seconds)
        for o in x: assert set(o) == set(x[0])
        d = dict()
        for k in x[0].keys():  
            d[k] = [o[k] for o in x]
        ## Tokenize with the pp_tokenizer and nli_tokenizer seperately 
        # pp tokenizer
        d_pp = copy.deepcopy(d)
        for k in ['orig_ids', 'attention_mask']: 
            d_pp[k] = d_pp.pop(f'{k}_pp_tknzr'); 
        d_pp['input_ids'] = d_pp['orig_ids']; d_pp.pop('orig_ids')
        batch_pp = self.pp_tokenizer.pad(d_pp, pad_to_multiple_of=1, return_tensors="pt")
        for k in ['input_ids', 'attention_mask']: batch_pp[f'{k}_pp_tknzr'] = batch_pp.pop(k)
        batch_pp['orig_ids_pp_tknzr'] = batch_pp.pop('input_ids_pp_tknzr')
        return_d = {**batch_pp}
        return return_d
        
    def add_vm_orig_score(self, batch):
        """Add the vm score of the orig text"""
        labels = tensor(batch['label'], device=self.vm_model.device)
        orig_probs,orig_preds = src.models.get_vm_probs_from_text(text=batch[self.text_field], vm_tokenizer=self.vm_tokenizer, vm_model=self.vm_model)
        batch['orig_truelabel_probs'] = gather(orig_probs,1, labels[:,None]).squeeze().cpu().tolist()
        batch['orig_vm_predclass'] = orig_preds.cpu().tolist()
        return batch
    
    def add_ld_orig_score(self, batch):
        """Add the language detection score of the orig text"""
        orig_probs,orig_preds = src.models.get_ld_probs_from_text(text=batch[self.text_field], ld_tokenizer=self.ld_tokenizer, ld_model=self.ld_model)
        batch['orig_ld_predclass'] = orig_preds.cpu().tolist()
        batch['orig_ld_probs'] = gather(orig_probs,1, orig_preds[:,None]).squeeze().cpu().tolist()
        return batch
    
    def prepare_data(self, dsd, get_dataloaders=False):
        dsd = dsd.map(self.add_idx, batched=True, with_indices=True)
        dsd = dsd.shuffle(seed=0)  # some datasets are ordered with all positive labels first, then all neutral... (don't want this)
        if self.args.n_shards > 0: 
            for k,v in dsd.items():  dsd[k] = v.shard(self.args.n_shards, 0, contiguous=True)  # contiguous to stop possible randomness of sharding
        for k,v in dsd.items(): dsd[k] = v.flatten_indices()
        dsd = dsd.map(self.add_vm_orig_score, batched=True,  batch_size=512)
        # Remove misclassified examples
        dsd = dsd.filter(lambda x: x['orig_vm_predclass']== x['label'])
        dsd = dsd.map(self.add_ld_orig_score,   batched=True,  batch_size=512)
        dsd = dsd.map(self.add_sts_embeddings,  batched=True,  batch_size=512)  # add STS score
        if gen_model_type(self.args.pp_name) == 't5':
            dsd = dsd.map(self.prep_input_for_t5_paraphraser,  batched=True,  fn_kwargs={'task': 'paraphrase'})  # preprocess raw text so pp model can read
            dsd = dsd.map(self.tokenize_fn,        batched=True,  fn_kwargs={'tokenizer': self.pp_tokenizer, 'use_prefix' : True})  # tokenize with pp_tokenizer, with prefix
        else: 
            dsd = dsd.map(self.tokenize_fn,        batched=True,  fn_kwargs={'tokenizer': self.pp_tokenizer, 'use_prefix' : False})  # tokenize with pp_tokenizer, without prefix
        dsd = dsd.rename_column("input_ids", "orig_ids_pp_tknzr")
        dsd = dsd.rename_column("attention_mask", "attention_mask_pp_tknzr")
        # add n_tokens & filter out examples that have more tokens than a threshold
        dsd = dsd.map(self.add_n_tokens,       batched=True, fn_kwargs={'field': "orig_ids_pp_tknzr"})  # add n_tokens
        if self.args.long_example_behaviour == 'remove':    dsd = dsd.filter(lambda x: x['n_tokens'] <= self.args.max_length_orig)
        if self.args.bucket_by_length: dsd = dsd.sort("n_tokens", reverse=True)  # sort by n_tokens (high to low), useful for cuda memory caching and reducing number of padding tokens
        assert dsd.column_names['train'] == dsd.column_names['validation'] == dsd.column_names['test']
        if get_dataloaders:
            dld = self.prep_dataloaders(dsd,args=self.args)
            self.update_dataset_info(dsd, dld)
            return dsd,dld
        else: 
            self.update_dataset_info(dsd)
            return dsd,None
    
    def prep_dataloaders(self, dsd, args): 
        if gen_model_type(args.pp_name) == 't5':
            dsd_numeric = dsd.remove_columns([self.text_field, f'{self.text_field}_with_prefix', 'lang'])
        elif gen_model_type(args.pp_name) == 'mt5':
            dsd_numeric = dsd.remove_columns([self.text_field, 'lang'])
        dld = self.get_dataloaders_dict(dsd=dsd_numeric, args=args, collate_fn=self.collate_fn)  # dict of data loaders that serve tokenized text
        return dld 
    
class GenFineTuningDataset(BaseDataset): 
    def __init__(self, args, tokenizer): 
        self.args = args
        self.tokenizer = tokenizer
        # Load preprocessed data
        # run tapaco_preprocessing.py first to create this dataset
        # comes already shuffled
        dsd = DatasetDict.load_from_disk("data/tapaco_processed")
        if self.args.n_shards > 0: 
            for k,v in dsd.items():  dsd[k] = v.shard(self.args.n_shards, 0, contiguous=True)  # contiguous to stop possible randomness of sharding

        dsd_numeric = dsd.remove_columns(['source', 'target'])
        self.dld = self.get_dataloaders_dict(dsd_numeric, args=self.args, collate_fn=self.collate_fn)  # dict of data loaders that serve tokenized text
        self.dsd = dsd 
        if self.args.disable_hf_caching: disable_caching()


    def collate_fn(self, x):
        """Collate function used by the DataLoader that serves tokenized data.
        x is a list (with length batch_size) of dicts. Keys should be the same across dicts.
        I guess an error is raised if not. """
        # check all keys are the same in the list. the assert is quick (~1e-5 seconds)
        for o in x: assert set(o) == set(x[0])
        d = dict()
        for k in x[0].keys():  
            d[k] = [o[k] for o in x]
        # the labels field doesn't get tokenized automatically, so we will rename to input_ids, tokenize, and then rename 
        d1 = dict(input_ids=d['input_ids'], attention_mask=d['attention_mask'])
        d2 = dict(input_ids=d['labels'])
        batch1 = self.tokenizer.pad(d1, padding=True, return_tensors='pt')
        batch2 = self.tokenizer.pad(d2, padding=True, return_tensors='pt')
        batch1['labels'] =  batch2['input_ids']
        return batch1 

class VictimFineTuningDataset(BaseDataset): 
    """This class just contains methods to finetune a victim model on a given dataset"""
    def __init__(self, args, vm_tokenizer): 
        self.ds_name = args.dataset_name
        for k, v in DS_INFO[self.ds_name].items(): setattr(self, k, v)
        self.args = args
        self.vm_tokenizer = vm_tokenizer
        if self.args.disable_hf_caching: disable_caching()
        dsd = self.load_data()
        self.dsd,self.dld = self.prepare_data(dsd)

    def tokenize_fn(self, batch, tokenizer):
        """Tokenize a batch of orig text using a tokenizer."""
        return tokenizer(batch[self.text_field], max_length=self.args.max_length_orig)

    def create_dataloaders(self, dsd): 
        dsd_numeric = dsd.remove_columns([self.text_field, 'lang'])
        dld = self.get_dataloaders_dict(dsd_numeric, args=self.args, collate_fn=self.collate_fn)  # dict of data loaders that serve tokenized text
        return dld

    def collate_fn(self, x):
        """Collate function used by the DataLoader that serves tokenized data.
        x is a list (with length batch_size) of dicts. Keys should be the same across dicts.
        I guess an error is raised if not. """
        # check all keys are the same in the list. the assert is quick (~1e-5 seconds)
        for o in x: assert set(o) == set(x[0])
        d = dict()
        for k in x[0].keys():  
            d[k] = [o[k] for o in x]
        batch_pp = self.vm_tokenizer.pad(d, return_tensors="pt", padding=True)
        return batch_pp 
    
    def prepare_data(self, dsd): 
        dsd = dsd.map(self.add_idx, batched=True, with_indices=True) 
        dsd = dsd.shuffle(seed=0)  # some datasets are ordered with all positive labels first, then all neutral... (don't want this)
        if self.args.n_shards > 0: 
            for k,v in dsd.items():  dsd[k] = v.shard(self.args.n_shards, 0, contiguous=True)  # contiguous to stop possible randomness of sharding
        dsd = dsd.map(self.tokenize_fn,  batched=True, fn_kwargs={'tokenizer': self.vm_tokenizer})  # tokenize with pp_tokenizer
        # add n_tokens & filter out examples that have more tokens than a threshold
        
        assert dsd.column_names['train'] == dsd.column_names['validation'] == dsd.column_names['test']
        dld = self.create_dataloaders(dsd)  # dict of data loaders that serve tokenized text
        self.update_dataset_info(dsd, dld)
        return dsd,dld
    
class BaselineDataset(BaseDataset): 
    def __init__(self, args, vm_tokenizer): 
        self.ds_name = args.dataset_name
        for k, v in DS_INFO[self.ds_name].items(): setattr(self, k, v)
        self.args = args
        self.vm_tokenizer = vm_tokenizer
        if self.args.disable_hf_caching: disable_caching()
        dsd = self.load_data()
        dsd = dsd['test']
        self.dsd = self.prepare_data(dsd)

    def tokenize_fn(self, batch, tokenizer):
        """Tokenize a batch of orig text using a tokenizer."""
        return tokenizer(batch[self.text_field], max_length=self.args.max_length_orig)

    def prepare_data(self, dsd): 
        dsd = dsd.map(self.add_idx, batched=True, with_indices=True) 
        dsd = dsd.shuffle(seed=0)  # some datasets are ordered with all positive labels first, then all neutral... (don't want this)
        # add n_tokens & filter out examples that have more tokens than a threshold
        dsd = dsd.map(self.tokenize_fn,   batched=True, fn_kwargs={'tokenizer': self.vm_tokenizer})  # tokenize with pp_tokenizer
        dsd = dsd.map(self.add_n_tokens,  batched=True, fn_kwargs={'field': "input_ids"})  # add n_tokens
        if self.args.long_example_behaviour == 'remove':    dsd = dsd.filter(lambda x: x['n_tokens'] <= self.args.max_length_orig)    
        # If doing dev, select only a subset 
        if self.args.n_examples > -1:  dsd = dsd.select(range(self.args.n_examples))
        dsd = dsd.map(function=lambda x: {"x": x[self.text_field], "y":  x["label"]})
        self.update_dataset_info(dsd)
        return dsd
    
    def update_dataset_info(self, ds): 
        d = dict()
        d[f"ds_test_len"] = len(ds)
        for i in range(self.num_labels):
            d[f"ds_test_class_{self.ID2LABEL[i]}"] =  len(ds.filter(lambda x: x['label'] == i))
        d['ds_num_labels']    = self.num_labels
        d['ds_text_field']    = self.text_field
        d['ds_label_field']   = self.label_field
        self.dataset_info = d
