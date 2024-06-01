import torch
from transformers import (AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer, MT5ForConditionalGeneration)
from sentence_transformers import SentenceTransformer
import logging
from src.victim_model import VictimModel
from src.gen_model import GenModel
from transformers import AutoConfig
from src.training_fns import get_device_string
from src.utils import gen_model_type

logger = logging.getLogger(__name__)

T5_PREFIX = {
    "paraphrase": "paraphrase: ", 
}

ALG_TOKENIZER_MAPPING = {
    'SentencePiece': ["T5", "Albert", "mt5"], 
    'WordPiece': ["Bert", "DistilBert", 'Electra'], 
    "BPE": ['Roberta', 'Deberta']
}
for k,v in ALG_TOKENIZER_MAPPING.items(): ALG_TOKENIZER_MAPPING[k] = [o + "TokenizerFast" for o in v]
TOKENIZER_ALG_MAPPING = dict()
for alg,tokenizer_l in ALG_TOKENIZER_MAPPING.items(): 
    for t in tokenizer_l: 
        TOKENIZER_ALG_MAPPING[t] = alg

        

##### LOAD MODELS #####
def load_reconstructed_mt5_tokenizer_and_model(args, path, ref_model=False): 
    with torch.device(get_device_string(args)):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = MT5ForConditionalGeneration.from_pretrained(path, local_files_only=True)
        model.resize_token_embeddings(len(tokenizer))
    if ref_model: 
        for i, (name, param) in enumerate(model.named_parameters()): param.requires_grad = False   # freeze 
        model.eval()
    else:         
        model.train()
    return tokenizer, model

def get_pp_tokenizer_and_model(model_name_or_path, args, ref_model=False):
    """Prepares pp model and tokenizer. or ref model and tokenizer. """
    if model_name_or_path[-5:] == ".ckpt":
        with torch.device(get_device_string(args)): 
            gen_module = GenModel.load_from_checkpoint(model_name_or_path) # , strict=False
            pp_config    = gen_module.config
            pp_tokenizer = gen_module.tokenizer
            pp_model     = gen_module.model
            pp_model.resize_token_embeddings(len(pp_tokenizer))
    else: 
        pp_config    = AutoConfig.from_pretrained(   model_name_or_path)
        pp_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, model_max_length=args.max_length_orig)
        gen_type = gen_model_type(model_name_or_path)
        if gen_type == "mt5" or gen_type == "t5":
            with torch.device(get_device_string(args)):
                pp_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, local_files_only=not args.download_models, force_download=args.download_models, config=pp_config)
            # For t5 there is a problem with different sizes of embedding vs vocab size. 
            # (and i assume with mt5 too?)
            # See https://github.com/huggingface/transformers/issues/4875
            pp_model.resize_token_embeddings(len(pp_tokenizer))
        else:       
            with torch.device(get_device_string(args)):
                pp_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, local_files_only= not args.download_models, config=pp_config, 
                    max_position_embeddings = args.max_length_orig + 10
                )
    if ref_model: 
        for i, (name, param) in enumerate(pp_model.named_parameters()): param.requires_grad = False   # freeze 
        pp_model.eval()
    else:          
        pp_model.train()
    return pp_tokenizer, pp_model

def get_vm_tokenizer_and_model(args):
    if args.vm_name[-5:] == ".ckpt":
        vm_module = VictimModel.load_from_checkpoint(args.vm_name)
        vm_config    = vm_module.config
        vm_tokenizer = vm_module.tokenizer
        vm_model     = vm_module.model
    else: 
        vm_config    = AutoConfig.from_pretrained(   args.vm_name)
        vm_tokenizer = AutoTokenizer.from_pretrained(args.vm_name)
        with torch.device(get_device_string(args)):
            vm_model = AutoModelForSequenceClassification.from_pretrained(args.vm_name, local_files_only=not args.download_models, force_download=args.download_models, config=vm_config)
    vm_model.eval()

    if args.freeze_vm_model: 
        for i, (name, param) in enumerate(vm_model.named_parameters()): param.requires_grad = False
    return vm_tokenizer, vm_model

def get_ld_tokenizer_and_model(args):
    ld_config    = AutoConfig.from_pretrained(   args.ld_name)
    ld_tokenizer = AutoTokenizer.from_pretrained(args.ld_name)
    ld_model     = AutoModelForSequenceClassification.from_pretrained(args.ld_name, local_files_only=not args.download_models,  config=ld_config)
    ld_model.eval()
    for i, (name, param) in enumerate(ld_model.named_parameters()): param.requires_grad = False
    return ld_tokenizer, ld_model

def get_cola_tokenizer_and_model(args):
    cola_config    = AutoConfig.from_pretrained(   args.cola_name)
    cola_tokenizer = AutoTokenizer.from_pretrained(args.cola_name)
    cola_model = AutoModelForSequenceClassification.from_pretrained(args.cola_name, local_files_only=not args.download_models, config=cola_config)
    cola_model.eval()
    for i, (name, param) in enumerate(cola_model.named_parameters()): param.requires_grad = False
    return cola_tokenizer, cola_model

def get_sts_model(args):
    with torch.device(get_device_string(args)):
        sts_model = SentenceTransformer(args.sts_name)
    for i, (name, param) in enumerate(sts_model.named_parameters()): param.requires_grad = False
    sts_model.eval()
    return sts_model

### INFERENCE 
def get_vm_probs_from_text(text, vm_tokenizer, vm_model, return_logits=False):
    """Get victim model predictions for a batch of text."""
    if vm_model.training: vm_model.eval()
    with torch.no_grad():
        tkns = vm_tokenizer(text, padding=True, truncation=True, pad_to_multiple_of=8, return_tensors="pt",max_length=256).to(vm_model.device)
        input_ids,attention_mask = tkns['input_ids'],tkns['attention_mask']
        start_id = vm_tokenizer.pad_token_id if vm_tokenizer.bos_token_id is None else vm_tokenizer.bos_token_id
        outputs = vm_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze()
        probs = torch.softmax(logits,1)
        preds = torch.argmax(logits, 1) 
    if return_logits: return logits 
    else:             return probs, preds

def get_vm_scores_from_vm_logits(labels, orig_truelabel_probs, vm_logits): 
    """vm_logits -> vm_scores"""
    vm_probs = vm_logits.softmax(axis=1)
    vm_predclass = torch.argmax(vm_probs, axis=1)
    vm_truelabel_probs   = torch.gather(vm_probs, 1, labels[:,None]).squeeze()
    vm_scores = orig_truelabel_probs - vm_truelabel_probs
    return dict(vm_predclass=vm_predclass, vm_truelabel_probs=vm_truelabel_probs, vm_scores=vm_scores)    

def get_ld_scores(orig_ld_probs, pp_ld_probs):  return orig_ld_probs - pp_ld_probs
    
def get_vm_scores_from_vm_logits_gumbel_sampled(labels, orig_truelabel_probs, vm_logits): 
    """vm_logits -> vm_scores for the case where vm_logits has gumbel samples"""
    gumbel_samples = vm_logits.shape[0]
    batch_size = vm_logits.shape[1]
    assert len(labels) == len(orig_truelabel_probs) == batch_size 
    vm_probs = vm_logits.softmax(axis=-1)
    vm_predclass = torch.argmax(vm_probs, axis=-1)
    assert vm_predclass.shape == (gumbel_samples, batch_size)
    labels_repeated = labels.repeat((gumbel_samples,1))
    assert labels_repeated.shape == (vm_probs.shape[0], vm_probs.shape[1]) == (gumbel_samples, batch_size)
    vm_truelabel_probs   = torch.gather(vm_probs, 2, labels_repeated[:,:,None]).squeeze()
    assert vm_truelabel_probs.shape == labels_repeated.shape
    orig_truelabel_probs_repeated = orig_truelabel_probs.repeat((gumbel_samples,1))
    assert orig_truelabel_probs_repeated.shape == vm_truelabel_probs.shape
    vm_scores = orig_truelabel_probs_repeated - vm_truelabel_probs
    return dict(vm_predclass=vm_predclass, vm_truelabel_probs=vm_truelabel_probs, vm_scores=vm_scores)    

def get_ld_scores_from_ld_logits_gumbel_sampled(orig_ld_predclass, orig_ld_probs, ld_logits):
    gumbel_samples = ld_logits.shape[0]
    batch_size = ld_logits.shape[1]
    ld_probs = ld_logits.softmax(axis=-1)
    ld_predclass_repeated = orig_ld_predclass.repeat((gumbel_samples,1))
    ld_predclass_pp_probs   = torch.gather(ld_probs, 2, ld_predclass_repeated[:,:,None]).squeeze()
    orig_ld_probs_repeated = orig_ld_probs.repeat((gumbel_samples,1))
    ld_scores = orig_ld_probs_repeated - ld_predclass_pp_probs
    return ld_scores

def get_nli_probs(orig_l, pp_l, nli_tokenizer, nli_model):
    inputs = nli_tokenizer(orig_l, pp_l, return_tensors="pt", padding=True, truncation=True).to(nli_model.device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
        probs = logits.softmax(1)
    return probs

def get_cola_probs(text, cola_tokenizer, cola_model):
    inputs = cola_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        logits = cola_model(**inputs).logits
        probs = logits.softmax(1)
    return probs

def get_ld_probs_from_text(text, ld_tokenizer, ld_model):
    """Get language detection scores for a batch of text."""
    inputs = ld_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(ld_model.device)
    with torch.no_grad():
        logits = ld_model(**inputs).logits
        probs = logits.softmax(1)
        preds = torch.argmax(logits, 1)
    return probs, preds

def get_logp(orig_ids, pp_ids, tokenizer, model):
    """model: one of pp_model, ref_model (same with tokenizer)"""
    start_id = tokenizer.pad_token_id if tokenizer.bos_token_id is None else tokenizer.bos_token_id
    decoder_start_token_ids = torch.tensor([start_id], device=model.device).repeat(len(orig_ids), 1)
    pp_ids = torch.cat([decoder_start_token_ids, pp_ids], 1)
    logprobs = []
    for i in range(pp_ids.shape[1] - 1):
        decoder_input_ids = pp_ids[:, 0:(i+1)]
        outputs = model(input_ids=orig_ids, decoder_input_ids=decoder_input_ids)
        token_logprobs = outputs.logits[:,i,:].log_softmax(1)
        pp_next_token_ids = pp_ids[:,i+1].unsqueeze(-1)
        pp_next_token_logprobs = torch.gather(token_logprobs, 1, pp_next_token_ids).detach().squeeze(-1)
        logprobs.append(pp_next_token_logprobs)
    logprobs = torch.stack(logprobs, 1)
    logprobs = torch.nan_to_num(logprobs, nan=None, posinf=None, neginf=-20) 
    logprobs = logprobs.clip(min=-20)
    attention_mask = model._prepare_attention_mask_for_generation(pp_ids[:,1:], tokenizer.pad_token_id, tokenizer.eos_token_id)
    logprobs = logprobs * attention_mask
    logprobs_sum = logprobs.sum(1)
    logprobs_normalised = logprobs_sum / attention_mask.sum(1)  # normalise for length of generated sequence
    return logprobs_normalised