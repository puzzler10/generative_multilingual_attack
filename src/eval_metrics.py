from lingua import Language, LanguageDetectorBuilder
import pandas as pd
import re 
import torch
import torch.nn as nn
import traceback
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from evaluate import load
import numpy as np
from itertools import product
from functools import partial

def format_sentence(x):
    """Get sentences in consistent format for evaluation. Perplexity and other fluency metrics 
    are very picky about this."""
    x = str(x)  # in case a number or a NaN comes through. 
    if len(x) == 0: return "sentence here."
    x = x.strip()  # Remove leading/trailing whitespaces
    try:
        x = x[0].upper() + x[1:]  # Ensure the first character is uppercase
        if re.search('[.!?]$', x): # Check if the sentence ends with a punctuation
            x = re.sub('\s+([.!?])$', r'\1', x)  # If it does, remove any spaces before the last punctuation mark
        else:
            x += '.'  # If it doesn't, add a full stop
    except: 
        return x
    return x

# Fluency score
class MBARTScorer:
    def __init__(self, device='cuda:0', max_length=512, checkpoint='facebook/mbart-large-50'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = MBart50TokenizerFast.from_pretrained(checkpoint)
        self.model = MBartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

        self.lang_codes = {
            "ar": "ar_AR",
            "de": "de_DE",
            "en": "en_XX",
            "es": "es_XX",
            "fr": "fr_XX",
        }

    def score(self, srcs, tgts, lang, batch_size=4):
        """ Score examples. only works for batch_size=1 for now.  """
        self.tokenizer.src_lang = self.lang_codes[lang]
        self.tokenizer.tgt_lang = self.lang_codes[lang]
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list
            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

def set_up_flu_scorer(device, model_name='facebook/mbart-large-50'):
    mbart_scorer = MBARTScorer(device=device, checkpoint=model_name)
    return mbart_scorer

def calc_flu_score(mbart_scorer, sentA, sentB, lang): 
    return mbart_scorer.score(srcs=[sentA], tgts=[sentB], lang=lang, batch_size=1)[0]

# Entailment score
def set_up_ent_model(device, model_name_or_path):        
    config_ent = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer_ent = AutoTokenizer.from_pretrained(model_name_or_path)
    model_ent = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config_ent)
    model_ent.eval()
    model_ent.to(device)
    if   model_name_or_path  == "microsoft/deberta-base-mnli"                        :   entail_label = 2
    elif model_name_or_path  == "howey/electra-small-mnli"                           :   entail_label = 0
    elif model_name_or_path  == "symanto/xlm-roberta-base-snli-mnli-anli-xnli"       :   entail_label = 0
    else:  entail_label = model_ent.config.label2id["entailment"]
    return model_ent, tokenizer_ent, entail_label

def calc_ent_score(model_ent, tokenizer_ent, entail_label, sentA, sentB):
    sentA, sentB = format_sentence(sentA), format_sentence(sentB)
    inputs = tokenizer_ent(sentA, sentB, return_tensors="pt", padding=True, truncation=True)
    inputs = inputs.to(model_ent.device)
    with torch.no_grad():
        logits = model_ent(**inputs).logits
        prob = logits.softmax(1)[0][entail_label].item()
    return prob
    
# Language similarity score
def set_up_lang_scorer():
    languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.ARABIC, Language.SPANISH]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    return detector

def calc_lang_score(detector, sentA, sentB):    
    orig_lang_result = detector.detect_language_of(sentA)
    pp_confidence_values = detector.compute_language_confidence_values(sentB)
    pp_origlang_confidence = [o.value for o in pp_confidence_values if o.language==orig_lang_result][0]
    return pp_origlang_confidence

# Similarity score 
def set_up_sim_scorer(): return load("bertscore")

def calc_sim_score(bertscore, bertscorer_model_name, sentA, sentB, device='cuda:0'): 
    return bertscore.compute(predictions=[sentA], references=[sentB],  model_type=bertscorer_model_name,
                              use_fast_tokenizer=True, device=device)['f1'][0]

## VSR threhsolds 
def get_vsr_fn_d(): 
    def threshold_fn(flu, ls, sim, f_thresh, l_thresh, s_thresh):
        return (flu > f_thresh and ls > l_thresh and sim > s_thresh)
    flu_thresholds = [-11, -12] # -7, -8, -9, -10, 
    langscore_thresholds = [0.50]
    sim_thresholds = [0.60] # , 0.70, 0.75
    fn_d = {}
    for f, l, s in product(flu_thresholds, langscore_thresholds, sim_thresholds):
        key = f'vsr_f{abs(f)}_l{int(l*100)}_s{int(s*100)}'
        fn_d[key] = partial(threshold_fn, f_thresh=f, l_thresh=l, s_thresh=s)
    return fn_d 

def get_vsr_d(df):
    fn_d = get_vsr_fn_d()
    results_d = {}
    flip = df['label_flip'].values
    for name, fn in fn_d.items():
        x = df.apply(lambda x: fn(flu=x.flu, ls=x.langscore, sim=x.sim), axis=1)
        df[name] = x * flip
        results_d[name] = np.mean(x * flip)
    return results_d

def set_up_all_scorers(device): 
    mbart_scorer =  set_up_flu_scorer(device, model_name='facebook/mbart-large-50')
    lang_scorer = set_up_lang_scorer()
    bertscorer_model_name = 'bert-base-multilingual-cased'
    bertscorer = set_up_sim_scorer()
    return {
        'flu': mbart_scorer,
        'langscore': lang_scorer,
        'sim': (bertscorer, bertscorer_model_name)
    }

def score_df(df, scorer_d, cname_orig='orig', cname_pp='pp', device='cuda:0'): 
    """df assumes columns: orig, pp, lang"""
    l=list()
    with torch.no_grad(): 
        for sentA,sentB,lang in zip(df[cname_orig].values, df[cname_pp].values, df['lang'].values):
            d = dict()
            if lang != 'ar':
                sentA, sentB = format_sentence(sentA), format_sentence(sentB)
            d['flu'] = calc_flu_score(scorer_d['flu'], sentA, sentB, lang)
            d['langscore'] = calc_lang_score(scorer_d['langscore'], sentA, sentB)
            d['sim'] = calc_sim_score(*scorer_d['sim'], sentA, sentB, device=device)
            l.append(d)
    results_df = pd.DataFrame(l)
    df_res = pd.concat([df, results_df], axis=1)
    return df_res