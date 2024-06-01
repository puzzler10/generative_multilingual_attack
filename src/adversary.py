import gc
import pickle
import os
import pandas as pd
import logging
from typing import List
from pytorch_lightning import LightningModule
import torch
import numpy as np
from src.eval_metrics import get_vsr_d
from transformers.utils.versions import require_version
from transformers import GenerationConfig 
from src.models import get_pp_tokenizer_and_model, get_sts_model, get_vm_tokenizer_and_model, get_vm_scores_from_vm_logits_gumbel_sampled,get_ld_tokenizer_and_model
from src.models import TOKENIZER_ALG_MAPPING, get_vm_probs_from_text, get_vm_scores_from_vm_logits,load_reconstructed_mt5_tokenizer_and_model, get_ld_probs_from_text
from src.models import get_logp, get_ld_scores_from_ld_logits_gumbel_sampled, get_ld_scores
from src.training_fns import args_havent_changed
from sentence_transformers.util import pytorch_cos_sim
from src.dataset_prep import DS_INFO 
from src.utils import * 
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from transformers.optimization import Adafactor, AdamW
from wandb import Histogram
from peft import LoraConfig, get_peft_model,TaskType

TOKEN_MAPPING_ARGS = ['pp_name', 'vm_name', 'sts_name', 'ld_name', 'lang', 'model_name_or_path']


class DebugAdversaryMixin:
    def __init__(self): 
        self.debug = in_debug_mode() 
        print("mixin init is running")
        if not self.debug:  # disable all debug methods by replacing them with do_nothing
            for attr_name in [o for o in dir(self) if o.startswith("debug")]:
                attr_value = getattr(self, attr_name)
                if callable(attr_value) and attr_name.startswith("debug"):
                    setattr(self, attr_name, self._do_nothing)

    def _do_nothing(self, *args, **kwargs): pass 

    def debug_check_models_tokenizers(self):
        """Checks that the models and tokenizers are set up properly."""
        ## Verify the different methods of calculating vocab and emb size give the same answer
        # NOTE: these asserts don't hold. 
        # The embedding matrix is size 32128 and the tokenizer vocab size is 32100. 
        # See https://github.com/huggingface/transformers/issues/4875 for a discussion. 
        # assert self.vocab_size_pp == self.pp_tokenizer.vocab_size
        # assert self.vocab_size_vm == self.vm_tokenizer.vocab_size
        assert self.emb_size['pp']   == self.pp_model.get_input_embeddings().embedding_dim
        assert self.emb_size['vm']   == self.vm_model.get_input_embeddings().embedding_dim
        assert self.emb_size['ld']   == self.ld_model.get_input_embeddings().embedding_dim
        assert self.vocab_size['pp'] == self.vocab_size['ref']  # for now - can add later but will have to token map
        # Check victim model and dataset have same number of classes
        if hasattr(self.vm_model, "lm_head"): assert self.dataset_num_labels == self.vm_model.lm_head.out_features  # for t5conditionalgeneration models 
        else:                                 assert self.dataset_num_labels == self.vm_model.num_labels

    def debug_forward_checks_and_assigns(self, batch, pp_logits, pp_ids, return_d): 
        ### Checks and sets during forward pass
        # Check batch size doesn't change after generation 
        assert pp_logits.shape == (self.batch_size, self.batch_len_pp_ids_pp_tknzr - 1, self.vocab_size['pp'])
        assert batch['orig_ids_pp_tknzr'].shape[0] == pp_ids.shape[0]
        # Store pp text if needed
        self.pp_text = self.pp_tokenizer.batch_decode(pp_ids, skip_special_tokens=True)
        return_d['pp_text'] = self.pp_text
        return return_d
    
    def log_forward(self, batch): 
        self.log_dict({
            "batch_size":                          float(self.batch_size), 
            "batch_len_orig_ids_pp_tknzr":         float(batch['orig_ids_pp_tknzr'].shape[1]), 
            "batch_len_pp_ids_pp_tknzr":           float(self.batch_len_pp_ids_pp_tknzr), 
        }, on_step=True)

    def debug_forward_print_text(self,batch, pp_ids): 
        print("ORIG")
        print(self.pp_tokenizer.convert_ids_to_tokens(batch['orig_ids_pp_tknzr'][0]))
        print(self.pp_tokenizer.convert_ids_to_tokens(batch['orig_ids_pp_tknzr'][1]))
        print("PP")
        print(self.pp_tokenizer.convert_ids_to_tokens(pp_ids[0]))
        print(self.pp_tokenizer.convert_ids_to_tokens(pp_ids[1]))

    def debug_check_pp_logits_nan(self, pp_logits):
        if torch.any(torch.isnan(pp_logits)): 
            raise Exception('we are generating nans')

    def debug_check_gumbel_softmax(self, gumbel_probs): 
        assert gumbel_probs.shape == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr - 1, self.vocab_size['pp'])

    def debug_weighted_emb_checks(self, gumbel_probs, weighted_emb_all, model_name): 
        assert gumbel_probs.shape == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr - 1, self.vocab_size['pp'])
        assert weighted_emb_all.shape    == (self.num_gumbel_samples, self.batch_size, (self.batch_len_pp_ids_pp_tknzr - 1), self.emb_size[model_name])

    def debug_weighted_emb_loop_checks(self, weights_reshaped, model_name, weighted_emb, weighted_emb_2d): 
        assert weights_reshaped.shape == (self.batch_size * (self.batch_len_pp_ids_pp_tknzr - 1), self.vocab_size['pp'])
        assert self.emb[model_name].shape      == (self.vocab_size[model_name], self.emb_size[model_name])
        assert weighted_emb.shape    == (self.batch_size * (self.batch_len_pp_ids_pp_tknzr - 1), self.emb_size[model_name] )
        assert weighted_emb_2d.shape    == (self.batch_size, (self.batch_len_pp_ids_pp_tknzr - 1), self.emb_size[model_name])

    def debug_prepare_model_inputs(self, pp_ids, weighted_emb, model_name,inputs_embeds, attention_mask): 
        assert pp_ids.shape       == (self.batch_size, self.batch_len_pp_ids_pp_tknzr)
        assert weighted_emb.shape == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr - 1, self.emb_size[model_name])
        assert inputs_embeds.shape == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr, self.emb_size[model_name])
        assert attention_mask.shape == (inputs_embeds.shape[1],inputs_embeds.shape[2])

    def log_weighted_emb_postprocessing(self,model_name,  weighted_emb_before_change_len, inputs_embeds): 
        self.log_dict({
            f"weighted_emb_length_{model_name}_before_postprocessing":       float(weighted_emb_before_change_len),
            f"weighted_emb_length_{model_name}_after_postprocessing":        float(inputs_embeds.shape[2]),
        }, on_step=True)     

    def debug_get_vm_logits_from_inputs_embeds(self, inputs_embeds, attention_mask, vm_logits, inputs_rep, attention_rep):
        assert inputs_embeds.shape  == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr, self.emb_size['vm'])
        assert attention_mask.shape == (self.batch_size, self.batch_len_pp_ids_pp_tknzr)
        assert inputs_rep.shape == (self.num_gumbel_samples * self.batch_size, inputs_embeds.shape[2],inputs_embeds.shape[3])
        assert attention_rep.shape == (self.num_gumbel_samples * self.batch_size, attention_mask.shape[1])
        assert vm_logits.shape      == (self.num_gumbel_samples, self.batch_size, self.dataset_num_labels)

    def debug_get_ld_logits_from_inputs_embeds(self, inputs_embeds, attention_mask, ld_logits, inputs_rep, attention_rep):
        assert inputs_embeds.shape  == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr, self.emb_size['ld'])
        assert attention_mask.shape == (self.batch_size, self.batch_len_pp_ids_pp_tknzr)
        assert inputs_rep.shape == (self.num_gumbel_samples * self.batch_size, inputs_embeds.shape[2],inputs_embeds.shape[3])
        assert attention_rep.shape == (self.num_gumbel_samples * self.batch_size, attention_mask.shape[1])
        assert ld_logits.shape      == (self.num_gumbel_samples, self.batch_size, self.ld_num_langs)

    def debug_get_sts_scores_and_diversity_score_from_inputs_embeds(self, inputs_embeds, attention_mask, orig_sts_embeddings, inputs_rep, attention_rep, orig_rep,
                                                                     pp_sts_embedding, sts_scores, sim_matrix_orig, sim_matrix_pp, diversity_score):
        assert orig_sts_embeddings.shape == (                         self.batch_size,                    self.emb_size['sts'])
        assert inputs_embeds.shape       == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr, self.emb_size['sts'])  # at this point CLS should have been included at front of embedding 
        assert attention_mask.shape      == (self.batch_size, self.batch_len_pp_ids_pp_tknzr)
        assert inputs_rep.shape == (self.num_gumbel_samples * self.batch_size, inputs_embeds.shape[2],inputs_embeds.shape[3])
        assert attention_rep.shape == (self.num_gumbel_samples * self.batch_size, attention_mask.shape[1])
        assert orig_rep.shape == (  self.num_gumbel_samples*  self.batch_size, self.emb_size['sts'])
        assert pp_sts_embedding.shape == orig_rep.shape
        assert sts_scores.shape == (self.num_gumbel_samples, self.batch_size)
        assert sim_matrix_orig.shape == sim_matrix_pp.shape == (self.batch_size, self.batch_size)
        assert diversity_score.shape == torch.Size([])

    def debug_get_kl_div_from_inputs_embeds(self, inputs_embeds, attention_mask_inputs, gumbel_probs, kl_divs_stacked, pp_probs_all, pp_logprobs_all): 
        assert inputs_embeds.shape         == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr,   self.emb_size['ref'])
        assert attention_mask_inputs.shape == (                         self.batch_size, self.batch_len_pp_ids_pp_tknzr)
        assert gumbel_probs.shape      ==     (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr-1, self.vocab_size['pp'])
        assert pp_probs_all.shape == pp_logprobs_all.shape == (self.num_gumbel_samples, self.batch_size, self.batch_len_pp_ids_pp_tknzr-1, self.vocab_size['pp'])
        assert kl_divs_stacked.shape == (self.num_gumbel_samples, self.batch_size)

    def debug_kl_div_inner_loop(self, i, decoder_inputs_embeds, ref_outputs, ref_logprobs, pp_probs, pp_logprobs, kl_div): 
        assert decoder_inputs_embeds.shape == (self.batch_size, i+1, self.emb_size['ref'])
        assert ref_outputs.logits.shape == (self.batch_size, i+1, self.vocab_size['ref'])
        assert ref_logprobs.shape == (self.batch_size, self.vocab_size['ref'])
        assert pp_probs.shape == pp_logprobs.shape == (self.batch_size, self.vocab_size['pp'])
        assert kl_div.shape == torch.Size([self.batch_size])

    def debug_kl_div_outer_loop(self, kl_divs, kl_divs_normalised):
        assert kl_divs.shape == (self.batch_size, self.batch_len_pp_ids_pp_tknzr-1)
        assert kl_divs_normalised.shape == torch.Size([self.batch_size])

    def debug_log_scores_histograms(self, vm_scores_clipped, sts_scores_clipped, ld_scores_clipped, kl_divs_clipped): 
        self.logger.experiment.log({"vm_scores_clipped":  Histogram(vm_scores_clipped.cpu().detach().numpy()), "step": self.global_step})
        self.logger.experiment.log({"sts_scores_clipped": Histogram(sts_scores_clipped.cpu().detach().numpy()), "step": self.global_step})
        self.logger.experiment.log({"ld_scores_clipped":  Histogram(ld_scores_clipped.cpu().detach().numpy()), "step": self.global_step})
        self.logger.experiment.log({"kl_divs_clipped":    Histogram(kl_divs_clipped.cpu().detach().numpy()), "step": self.global_step})

    def debug_log_diversity_component_histograms(self, sim_matrix_orig, sim_matrix_pp, sq_diff):
        sim_matrix_triu_orig = sim_matrix_orig[sim_matrix_orig.triu(diagonal=1) != 0].cpu().detach().numpy()
        sim_matrix_triu_pp   = sim_matrix_pp[  sim_matrix_pp.triu(diagonal=1)   != 0].cpu().detach().numpy()
        sim_matrix_sq_diff   = sq_diff[sq_diff != 0].cpu().detach().numpy()
        self.logger.experiment.log({"sim_matrix_triu_orig":  Histogram(sim_matrix_triu_orig), "step": self.global_step})
        self.logger.experiment.log({"sim_matrix_triu_pp": Histogram(sim_matrix_triu_pp), "step": self.global_step})
        self.logger.experiment.log({"sim_matrix_sq_diff": Histogram(sim_matrix_sq_diff), "step": self.global_step})

    def log_batch_losses(self, kl_penalty, diversity_penalty, loss_examples_mean, loss_batch):
        self.log("kl_penalty_batch",         kl_penalty.item(),          on_step=True)
        self.log("diversity_penalty_batch",  diversity_penalty.item(),   on_step=True)
        self.log("loss_examples_batch_mean", loss_examples_mean.item(),  on_step=True)
        self.log("loss_batch",               loss_batch.item(),          on_step=True)

    def training_step_add_forward_d(self, forward_d, return_d): 
        # detach values to save memory because we are just logging these values to a csv
        for k,v in forward_d.items(): 
            if type(v) is torch.Tensor:
                if v.grad_fn is not None: 
                    forward_d[k] = v.detach()
                forward_d[k] = forward_d[k].cpu()
        return_d = {**return_d, **forward_d}
        return return_d


class MultilingualWhiteboxAdversary(LightningModule, DebugAdversaryMixin): 
    def __init__(self, args):
        super().__init__()
        DebugAdversaryMixin.__init__(self)
        self.save_hyperparameters()  # Save arguments to hparams attribute.
        self.args = args
        self.debug = in_debug_mode() 
        self.adversary_info = dict()
        self.orig_cols_to_ignore = ['attention_mask_pp_tknzr', 'orig_ids_pp_tknzr', 'orig_sts_embeddings']
        self.dataset_num_labels = DS_INFO[args.dataset_name]['num_labels']
        self.eval_vm_threshold = (1 - (1.0/self.dataset_num_labels))
        if "reconstructed" in args.pp_name:
            # You can update this with your path
            path_mt5 = "/home/tproth/Data/model_checkpoints/multilingual_whitebox_finetuning/final/mt5_reconstructed"
            self.pp_tokenizer,   self.pp_model  = load_reconstructed_mt5_tokenizer_and_model(args, path_mt5)
            self.ref_tokenizer,  self.ref_model = load_reconstructed_mt5_tokenizer_and_model(args, path_mt5, ref_model=True)
        else:
            self.pp_tokenizer,   self.pp_model   = get_pp_tokenizer_and_model(model_name_or_path=args.pp_name, args=args)
            self.ref_tokenizer,  self.ref_model  = get_pp_tokenizer_and_model(model_name_or_path=args.ref_name, args=args, ref_model=True)

        if self.args.use_peft:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=16,
                target_modules=["q", "v"],
                lora_dropout=0.1,
                bias="none",
                task_type =TaskType.SEQ_2_SEQ_LM
            )
            self.pp_model = get_peft_model(self.pp_model, peft_config)
            self.pp_model.print_trainable_parameters()
            

        self.vm_tokenizer,   self.vm_model   = get_vm_tokenizer_and_model(args)
        self.ld_tokenizer,   self.ld_model    = get_ld_tokenizer_and_model(args)
        self.ld_num_langs =   self.ld_model.num_labels 

        self.sts_model    = get_sts_model(args)
        self.sts_base_model    = self.sts_model[0].auto_model
        self.sts_pooling_layer = self.sts_model[1]
        self.sts_tokenizer = self.sts_model.tokenizer

        if self.args.sts_name == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":
            # Seems to come up as a problem for this type of model. 
            self.sts_base_model.resize_token_embeddings(len(self.sts_tokenizer))

        # Store conveniently 
        self.tokenizers = {
            "pp":  self.pp_tokenizer, 
            "ref": self.ref_tokenizer,  
            "vm": self.vm_tokenizer, 
            "sts": self.sts_tokenizer, 
            "ld": self.ld_tokenizer, 
        }
        self.tokenizer_names = {k:v.__class__.__name__ for k,v in self.tokenizers.items()}
        self.tokenizer_algs = {k:TOKENIZER_ALG_MAPPING[v] for k,v in self.tokenizer_names.items()}
        self.models = { 
            "pp": self.pp_model, 
            'ref': self.ref_model,
            "vm": self.vm_model, 
            "sts": self.sts_base_model, 
            "ld": self.ld_model, 
        }
        for k, model in self.models.items(): 
            if k != "pp" and model.training: model.eval()

        # MT5 model uses a T5 tokenizer too
        if not 'T5Tokenizer' in self.tokenizer_names['pp'] : raise Exception("For now, paraphaser must be a mt5 or t5 model.")

        ## Embeddings and sizes 
        self.emb,self.vocab_size,self.emb_size = dict(),dict(),dict()
        for model_name in self.models.keys(): 
            self.emb[model_name] = self.models[model_name].get_input_embeddings().weight
            self.vocab_size[model_name],  self.emb_size[model_name]  = self.emb[model_name].size()

        # Check if the vocab's of pp and vm are matched or not. 
        self.matched_vocab_sizes = {k:v.vocab==self.pp_tokenizer.vocab for k,v in self.tokenizers.items() }
        self.debug_check_models_tokenizers()

        ## Get token mapping
        # read from cache if available
        if args_havent_changed(args, TOKEN_MAPPING_ARGS) and os.path.isfile(f'{args.cache_dir}/token_mapping_cached.pickle'):
            with open(f'{args.cache_dir}/token_mapping_cached.pickle', 'rb') as handle: 
                print("Loading token mapping from cache")
                token_mapping,self.df_token_mapping, self.adversary_info = pickle.load(handle)
                for model_name in token_mapping.keys(): 
                    self.register_buffer(f"token_mapping['{model_name}']",  token_mapping[model_name])
        else: # no cache available
            print(f"No token mapping cache available.")
            token_mapping,self.df_token_mapping,self.token_mapping_stats = dict(),dict(),dict()
            for model_name in [o for o in self.models.keys() if o not in ['pp']]: 
                if not self.matched_vocab_sizes[model_name]: 
                    token_mapping[model_name],self.df_token_mapping[model_name] = self._get_token_mapping_sparse_matrix(self.tokenizers[model_name], model_name)
                    # register these so they are moved to device, saved, cast to right type, etc 
                    # register buffer used for things that are model state but not trainable parameter
                    self.register_buffer(f"token_mapping['{model_name}']",  token_mapping[model_name])
                self.adversary_info.update(self.token_mapping_stats) 
            with open(f'{args.cache_dir}/token_mapping_cached.pickle', 'wb') as handle: 
                pickle.dump((token_mapping,self.df_token_mapping, self.adversary_info), handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.adversary_info.update({  # for logging
                "vocab_size": self.vocab_size, 
                "emb_size":   self.emb_size,
                "matched_vocab_sizes": self.matched_vocab_sizes
        })

        # Generation parameters
        self.gen_params_train = dict(num_return_sequences=1, num_beams=1, do_sample=False, 
                                     return_dict_in_generate=True, output_scores=True)
        self.gen_config_train = GenerationConfig(**self.gen_params_train)
        self.gen_config_eval_d = dict()
        for k,v in self.args.gen_settings.items(): 
            self.gen_config_eval_d[k] = GenerationConfig(**v)
        ## Data stores 
        # training is for training
        self.df_l = dict(training=[],train=[],validation=[],test=[])
        # testing 
        self.test_results_d = dict()
        for eval_setting in self.gen_config_eval_d.keys():
            if eval_setting != 'val': self.test_results_d[eval_setting] = []
        self.asr_d = dict()

    def _get_token_mapping_sparse_matrix(self, tokenizer, model_name): 
        """Get token mapping matrix between pp_model and tokenizer specified."""
        ### Identify the VM tokenizer type and the subword string
        pp_space_str = "▁"  # denotes space for (m)T5. not actually a uscore - see: ▁_▁_▁_
        tokenizer_alg = self.tokenizer_algs[model_name]
        if tokenizer_alg == "WordPiece":   tkn_continue_str = "##"  # "##" token denotes word continuation 

        ## MATCHING
        # Get tokens from vocab and put into dataframe
        V_pp,V_tk = self.pp_tokenizer.vocab,tokenizer.vocab
        V_pp_df = pd.DataFrame({'pp_tkn':V_pp.keys(), 'pp_idx':V_pp.values()})
        V_tk_df = pd.DataFrame({'tk_tkn':V_tk.keys(), 'tk_idx':V_tk.values()})
        V_pp_df['pp_idx'] = V_pp_df['pp_idx'].astype('str')
        V_tk_df['tk_idx'] = V_tk_df['tk_idx'].astype('str')
        # Create flags for if there is any special tokens. Then strip them out
        V_pp_df['has_pp_space_str']    = V_pp_df['pp_tkn'].map(lambda x: pp_space_str in x)
        V_tk_df['has_tk_continue_str'] = V_tk_df['tk_tkn'].map(lambda x: tkn_continue_str in x)
        V_pp_df['pp_tkn'] = V_pp_df['pp_tkn'].map(lambda x: x.replace(pp_space_str, ''))
        V_tk_df['tk_tkn'] = V_tk_df['tk_tkn'].map(lambda x: x.replace(tkn_continue_str, ''))
        
        # First: special token matches
        special_mapping = dict()
        for k in self.pp_tokenizer.additional_special_tokens_ids: special_mapping[k] = tokenizer.unk_token_id  # tokens like <extra_id_2>
        if tokenizer_alg == "WordPiece":
            special_mapping.update({
                self.pp_tokenizer.pad_token_id: tokenizer.pad_token_id,
                self.pp_tokenizer.eos_token_id: tokenizer.sep_token_id,
                self.pp_tokenizer.unk_token_id: tokenizer.unk_token_id,
                # 3: 1517 # 3 is space str ▁, 1517 is em dash. there is no space character in wordpiece tokenizer so picking a randomish punctuation char
            })
        else:
             raise Exception("only WordPiece token algorithm supported so far for modules. need to configure the mapping including the special token map.")
        pp_special_ids = [str(o) for o in special_mapping.keys()]
        df_special_match = V_pp_df.query('pp_idx in @pp_special_ids').copy(deep=True)
        df_special_match['tk_idx'] = df_special_match['pp_idx'].map(lambda x: str(special_mapping[int(x)]))
        df_special_match = df_special_match.merge(right=V_tk_df[['tk_idx','tk_tkn']], on='tk_idx', how='left')
        df_special_match['has_pp_space_str'] = False
        df_special_match['has_tk_continue_str'] = False
        df_special_match['match_type'] = "special_tkn" 

        # CASE 1 matches:  _ with pp vocab (ie start of word), and no ## for tk (ie not middle of word)
        # e.g.: pp "_shift" to tk "shift"
        df_case1_match = pd.merge(V_pp_df.query('has_pp_space_str==True'), V_tk_df.query('has_tk_continue_str==False'), left_on='pp_tkn', right_on='tk_tkn', how='inner')
        df_case1_match['match_type'] = "case1" 

        # CASE 2 matches: no _ with pp vocab (ie not start of word), and ## for tk (ie middle of word)
        # e.g.: pp "shift" to tk "##shift"
        df_case2_match = pd.merge(V_pp_df.query('has_pp_space_str==False'), V_tk_df.query('has_tk_continue_str==True'), left_on='pp_tkn', right_on='tk_tkn', how='inner')
        df_case2_match['match_type'] = "case2" 
        
        # Tokeniser matches
        # Identify unmatched tokens, put them through tk tokenizer and get mapping
        matched_pp_idx = list(df_special_match['pp_idx']) + list(df_case1_match['pp_idx']) + list(df_case2_match['pp_idx']) 
        unmatched_pp_tokens_idxs = V_pp_df.query('pp_idx not in @matched_pp_idx')[['pp_tkn','pp_idx']].values
        l = []
        for pp_tkn,pp_idx in unmatched_pp_tokens_idxs:   
            d = dict(pp_tkn=pp_tkn, pp_idx=pp_idx, tk_idx=tokenizer(pp_tkn, add_special_tokens=False)['input_ids'])  # get tokenizer mapping of each token
            l.append(d)
        # Convert to dataframe, merge, and add weights
        df_tokeniser_match = unpack_nested_lists_in_df(pd.DataFrame(l), scalar_cols=['pp_tkn','pp_idx'])
        df_tokeniser_match['match_type']= 'tokeniser'
        df_tokeniser_match['pp_idx'] = df_tokeniser_match['pp_idx'].astype('str')
        df_tokeniser_match['tk_idx'] = df_tokeniser_match['tk_idx'].astype('str')
        df_tokeniser_match = df_tokeniser_match.merge(V_tk_df, on='tk_idx').merge(V_pp_df[['pp_idx', 'has_pp_space_str']], on='pp_idx')

        ## Concat together all match types to make final df of matches
        df_all = pd.concat([df_special_match, df_case1_match, df_case2_match, df_tokeniser_match])
        df_all['tk_idx'] = df_all['tk_idx'].astype('str')
        # Check no duplicates
        sizes =  df_all.groupby(['pp_idx'])['match_type'].nunique().to_frame('n_match_types').reset_index()
        n_duplicate_matches = len(sizes.query('n_match_types>1'))

        # Handle unmatched tokens. These are edge cases like \xad that didn't work for some reason 
        df_unmatched = V_pp_df.merge(df_all.pp_idx.to_frame('pp_idx_matched').drop_duplicates(), left_on='pp_idx',right_on='pp_idx_matched',  how='left')
        df_unmatched['unmatched'] = df_unmatched['pp_idx_matched'].isna()
        df_unmatched = df_unmatched.query('unmatched==True').copy(deep=True)
        # General workaround is to map to UNK token for these unmatched tokens
        df_unmatched['tk_tkn'] =  tokenizer.unk_token
        df_unmatched['tk_idx'] = tokenizer.unk_token_id
        df_unmatched['has_tk_continue_str'] = False
        df_unmatched['match_type'] = "unmatched" 
        df_all = pd.concat([df_all, df_unmatched])

        # Get weights
        df_all = df_all.join(1/df_all.groupby('pp_idx')['tk_idx'].size(), on='pp_idx', rsuffix='_r').rename(columns={'tk_idx_r': 'weight'})
        # Check that weights sum to 1
        weight_sums = df_all[['pp_idx','weight']].groupby('pp_idx')['weight'].sum().value_counts()
        assert weight_sums.index[0] == 1 and len(weight_sums) == 1
        # Check no duplicates in pp_idx and match_type
        assert max(df_all[['pp_idx', 'match_type']].drop_duplicates().groupby('pp_idx').size().value_counts().index) == 1
        
        # Create sparse mapping matrix 
        token_mapping = torch.sparse_coo_tensor(
            indices=np.array([df_all['pp_idx'].values.astype('int32'), df_all['tk_idx'].values.astype('int32')]), 
            values=df_all['weight'].values,
            size=(len(V_pp), len(V_tk)), device=self.device, requires_grad=False
        )
        # coalesce sums up the weights for all [pp_idx, tk_idx] duplicates. needed because sometimes you have duplicate tokens in a mapping 
        # for example, "...hello" might map to ".",".",".","hello", and weights in df_all will be 0.25 for each "." but what we really 
        # want is 0.75 weight for one "."
        # coalesce does the sum for us. if you don't do coalesce, then you get a vector of [0.25, 0.25, 0.25] at the corresponding entry 
        #  for ["...hello","."], instead of the correct scalar of 0.75. 
        token_mapping = token_mapping.coalesce()
        
        # Save stats and dataframe for logging
        df_all = df_all.drop(columns=['unmatched', 'pp_idx_matched'])
        self.token_mapping_stats[model_name] = {
            f"token_matches_unmatched_tokens" : len(df_unmatched),
            f"token_matches_duplicate" :n_duplicate_matches,
            f"token_matches_tokeniser" :len(df_tokeniser_match.pp_idx.unique()),
            f"token_matches | pp: _x -> tk: x" : len(df_case1_match),
            f"token_matches | pp: x -> tk: ##x" : len(df_case2_match),
            f"token_matches_special" : len(df_special_match),
            f"token_mapping_shape": token_mapping.shape
        }
        return token_mapping, df_all

    def forward(self, batch):
        """Prediction/inference only"""
        # Generate, ch
        self.batch_size = batch['orig_ids_pp_tknzr'].shape[0]
        pp_logits,pp_ids = self._generate_pp(batch, gen_config=self.gen_config_train) 
        self.batch_len_pp_ids_pp_tknzr   =  pp_ids.shape[1]
        self.debug_check_pp_logits_nan(pp_logits)

        # Get gumbel samples
        gumbel_probs = self._get_gumbel_samples(pp_logits)
        weighted_emb,model_inputs = dict(),dict()
        for model_name in ['vm','sts', 'ld', 'ref']:     
            weighted_emb[model_name] = self._construct_weighted_emb(gumbel_probs, model_name)
        for model_name in ['vm', 'sts', 'ld', 'ref']: 
            model_inputs[model_name] = self._prepare_model_inputs(pp_ids, weighted_emb[model_name], model_name)

        vm_logits = self._get_vm_logits_from_inputs_embeds(model_inputs['vm'])
        vm_scores_d = get_vm_scores_from_vm_logits_gumbel_sampled(labels=batch['label'], orig_truelabel_probs=batch['orig_truelabel_probs'], vm_logits=vm_logits)
        sts_scores, diversity_score = self._get_sts_scores_and_diversity_score_from_inputs_embeds(batch['orig_sts_embeddings'], model_inputs['sts'])
        kl_divs = self._get_kl_div_from_inputs_embeds(orig_ids=batch['orig_ids_pp_tknzr'], orig_attention_mask=batch['attention_mask_pp_tknzr'],
                                                      gumbel_probs=gumbel_probs, inputs=model_inputs['ref'])
        ld_logits = self._get_ld_logits_from_inputs_embeds(model_inputs['ld'])  # TODO
        ld_scores = get_ld_scores_from_ld_logits_gumbel_sampled(orig_ld_predclass=batch['orig_ld_predclass'], orig_ld_probs=batch['orig_ld_probs'], ld_logits=ld_logits)

        return_d =  {**vm_scores_d, 'sts_scores': sts_scores, 'ld_scores': ld_scores, 'kl_divs': kl_divs, 'diversity_score': diversity_score}
        self.log_forward(batch)
        if self.debug: return_d = self.debug_forward_checks_and_assigns(batch, pp_logits, pp_ids, return_d)
        return return_d

    def forward_eval(self, batch, eval_setting):  
        """assume no labels with this function."""
        with torch.no_grad():
            gen_config = self.gen_config_eval_d[eval_setting]
            self.batch_size_eval = batch['orig_ids_pp_tknzr'].shape[0]
            # Generate paraphrases
            _,pp_ids = self._generate_pp(batch, gen_config=gen_config) 
            pp_text = self.pp_tokenizer.batch_decode(pp_ids, skip_special_tokens=True)       
            n_pp = gen_config.num_return_sequences    
            
            # VM scores    
            vm_logits = get_vm_probs_from_text(text=pp_text, vm_tokenizer=self.vm_tokenizer, vm_model=self.vm_model, return_logits=True)
            def nest_tensor(x): 
                # the docs promise that we get exactly num_return_sequences*batch_size_eval paraphrases back, so we can nest them like this. 
                return x.reshape(int(x.shape[0]/n_pp), n_pp,  x.shape[1])
            vm_logits_nested = nest_tensor(vm_logits) 

            # STS scores
            pp_sts_embeddings = self.sts_model.encode(pp_text, convert_to_tensor=True, device=self.device, show_progress_bar=False)
            pp_sts_embeddings_nested = pp_sts_embeddings.reshape(int(pp_sts_embeddings.shape[0]/n_pp), n_pp,  pp_sts_embeddings.shape[1])
            sts_scores_t = torch.stack(list(map(lambda orig, pp: pytorch_cos_sim(orig, pp), batch['orig_sts_embeddings'], pp_sts_embeddings_nested))).squeeze()
            sts_scores = sts_scores_t.cpu().tolist()            
           
            # LD scores 
            ld_probs,_ = get_ld_probs_from_text(text=pp_text, ld_tokenizer=self.ld_tokenizer, ld_model=self.ld_model)
            orig_ld_predclass_repeated = batch['orig_ld_predclass'].repeat_interleave(repeats=n_pp,dim=0)
            ld_probs = torch.gather(ld_probs, 1,  orig_ld_predclass_repeated[:,None]).squeeze()
            orig_ld_probs_repeated = batch['orig_ld_probs'].repeat_interleave(repeats=n_pp,dim=0)
            ld_scores = get_ld_scores(orig_ld_probs=orig_ld_probs_repeated, pp_ld_probs=ld_probs)
            ld_scores_nested = [ld_scores[i:i+n_pp].cpu().tolist()  for i in range(0, len(ld_scores), n_pp)]

            # KL_DIV 
            orig_ids_repeated = batch['orig_ids_pp_tknzr'].repeat_interleave(repeats=n_pp,dim=0)
            with torch.no_grad(): 
                pp_logp  = get_logp(orig_ids_repeated, pp_ids, self.tokenizers['pp'],  self.models['pp'])
                ref_logp = get_logp(orig_ids_repeated, pp_ids, self.tokenizers['ref'], self.models['ref'])
                kl_div = pp_logp - ref_logp
            def nest(x): return x.reshape(int(x.shape[0]/n_pp), n_pp)
            pp_logp_nested_t,ref_logp_nested_t,kl_div_nested_t = nest(pp_logp), nest(ref_logp), nest(kl_div)
            pp_logp_nested,ref_logp_nested,kl_div_nested = pp_logp_nested_t.cpu().tolist(), ref_logp_nested_t.cpu().tolist(), kl_div_nested_t.cpu().tolist()
            # text
            pp_text_nested = [pp_text[i:i+n_pp] for i in range(0, len(pp_text), n_pp)]  # put paraphrases in nested lists
            
            if self.debug:
                assert vm_logits_nested.shape == (self.batch_size_eval, n_pp, self.dataset_num_labels)
                assert vm_logits.shape == (self.batch_size_eval * n_pp , self.dataset_num_labels)
                assert pp_sts_embeddings.shape == (self.batch_size_eval * n_pp, self.emb_size['sts'])
                assert pp_sts_embeddings_nested.shape == (self.batch_size_eval, n_pp, self.emb_size['sts'])
                if n_pp == 1: 
                    assert sts_scores_t.shape == (self.batch_size_eval,)
                else:         
                    assert sts_scores_t.shape == (self.batch_size_eval, n_pp)
                assert kl_div.shape == torch.Size([self.batch_size_eval * n_pp])
                assert orig_ids_repeated.shape == (self.batch_size_eval * n_pp, batch['orig_ids_pp_tknzr'].shape[1])
                assert pp_logp_nested_t.shape == ref_logp_nested_t.shape == kl_div_nested_t.shape == torch.Size([self.batch_size_eval, n_pp])
                assert len(pp_text_nested) == self.batch_size_eval
                assert ld_probs.shape == orig_ld_probs_repeated.shape
                assert all([len(l)==n_pp for l in ld_scores_nested])
                assert all([len(l)==n_pp for l in pp_text_nested])

        return {'pp_text_nested': pp_text_nested, 'vm_logits_nested': vm_logits_nested, 'sts_scores': sts_scores, 'ld_scores': ld_scores_nested,
                'pp_logp': pp_logp_nested, 'ref_logp': ref_logp_nested, 'kl_divs': kl_div_nested}

    def  _generate_pp(self, batch, gen_config): 
        """Generate token-transition logits and paraphrase ids, given a batch and the generation parameters"""
        inputs = {'input_ids': batch['orig_ids_pp_tknzr'], 'attention_mask': batch['attention_mask_pp_tknzr']}
        input_len = batch['orig_ids_pp_tknzr'].shape[1]
        gen_config.min_new_tokens = int(max(0, input_len - 2 - np.floor(input_len/4)))
        gen_config.max_new_tokens = input_len + 2
        pp_output = self.pp_model.generate(**inputs, generation_config=gen_config)
        pp_logits,pp_ids = torch.stack(pp_output.scores, dim=1),pp_output.sequences
        return pp_logits,pp_ids

    def _get_gumbel_samples(self, pp_logits): 
        ## Take B samples from the gumbel_softmax distribution to approximate the softmax over log_coeffs
        # This uses a default Tau temperature of 1, and uses soft probabilities rather than the 
        #   harder one-hot (hard set to False)
        self.num_gumbel_samples = self.args.num_gumbel_samples
        # gumbel softmax returns probs not logits
        # tau -> 0 makes it much harder (closer to one-hot)
        # tau -> inf makes it much softer (closer to uniform)
        # change dtype of gumbel_probs to float32


        def gumbel_softmax_no_nans(logits, tau=1, hard=False, eps=1e-10, dim=-1):
            r"""
            a modification on the torch gumbel softmax to avoid nans 
            https://gist.github.com/GongXinyuu/3536da55639bd9bfdd5a905ebf3ab88e 
            """
            def _gen_gumbels():
                gumbels = -torch.empty_like(logits).exponential_().log()
                if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum(): gumbels = _gen_gumbels() # to avoid zero in exp output
                return gumbels
            gumbels = _gen_gumbels()  # ~Gumbel(0,1)
            gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
            y_soft = gumbels.softmax(dim)
            if hard: # Straight through.
                index = y_soft.max(dim, keepdim=True)[1]
                y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
                ret = y_hard - y_soft.detach() + y_soft
            else: # Reparametrization trick.
                ret = y_soft
            return ret
        
        gumbel_probs = gumbel_softmax_no_nans(pp_logits.repeat(self.num_gumbel_samples, 1,1,1), tau=self.args.gumbel_tau, hard=False)
        # Add a small amount to handle non-zeros
        gumbel_probs = gumbel_probs + 1e-12
        self.debug_check_gumbel_softmax(gumbel_probs)
        return gumbel_probs

    def _construct_weighted_emb(self, gumbel_probs, model_name): 
        """Construct weighted embeddings (i.e. \"expected\" embeddings) from the per-token probabilities given in `pp_logits`. """
        weighted_emb_l = []
        for b in range(self.num_gumbel_samples): 
            weights = gumbel_probs[b, :, :, :]
            weights_reshaped = weights.view(-1, self.vocab_size['pp'])
            if self.matched_vocab_sizes[model_name]:  
                weighted_emb = weights_reshaped.mm(self.emb[model_name])
            else:                
                # first argument to torch.sparse.mm has to be sparse, second can be sparse or dense. output: dense  
                # because of this we have to take transposes of everything and then take transpose of the final result 
                # same :  wemb  = weights @ token_map  @ emb           
                #         wembT = embT    @ token_mapT @ weightsT   
                tkn_map = self.__getattr__(f"token_mapping['{model_name}']")
                if self.debug: assert tkn_map.shape == (self.vocab_size['pp'], self.vocab_size[model_name])
                weighted_emb = (self.emb[model_name].double().t().mm(torch.sparse.mm(tkn_map.t(), weights_reshaped.t().double()))).t()
            weighted_emb_2d = weighted_emb.view(-1, (self.batch_len_pp_ids_pp_tknzr - 1), self.emb_size[model_name])
            self.debug_weighted_emb_loop_checks(weights_reshaped, model_name, weighted_emb, weighted_emb_2d)
            weighted_emb_l.append(weighted_emb_2d)
        weighted_emb_all = torch.stack(weighted_emb_l, dim=0)
        self.debug_weighted_emb_checks(gumbel_probs, weighted_emb_all, model_name)
        return weighted_emb_all

    def _prepare_model_inputs(self, pp_ids,  weighted_emb, model_name): 
        """Prepare the model inputs for component models. e.g. add CLS token for a bert model / wordpiece model.  """
        weighted_emb_before_change_len = weighted_emb.shape[2]
        tokenizer_name = self.tokenizer_names[model_name]
        if  TOKENIZER_ALG_MAPPING[tokenizer_name] == "SentencePiece": # used for ref model 
            # format for sentnece X:  [PAD] X [SEP]
            pad_emb = self.emb[model_name][self.tokenizers[model_name].pad_token_id,:].unsqueeze(0).repeat(self.num_gumbel_samples,self.batch_size,1,1).to(self.device)
            inputs_embeds = torch.concat([pad_emb, weighted_emb], dim=2) 
            pre_ids  = torch.tensor(self.tokenizers[model_name].pad_token_id).repeat(self.batch_size, 1).to(self.device) # Duplicate PAD across the batch size
            if self.debug: assert pad_emb.shape == (self.num_gumbel_samples, self.batch_size, 1, self.emb_size[model_name])
        elif TOKENIZER_ALG_MAPPING[tokenizer_name] == "WordPiece":  
            # Format for sentence X : [CLS] X [SEP] 
            # The last probailities in pp_logits are to calculate p(sep|pp) and this works out as a vector pretty close to SEP anyway. So we don't need to add it on at the end. 
            # You should end up with weighted_emb being equal in shape to pp_ids. 
            start_cls_emb = self.emb[model_name][self.tokenizers[model_name].cls_token_id,:].unsqueeze(0).repeat(self.num_gumbel_samples,self.batch_size,1,1).to(self.device)
            inputs_embeds = torch.concat([start_cls_emb, weighted_emb], dim=2) 
            pre_ids  = torch.tensor(self.tokenizers[model_name].cls_token_id).repeat(self.batch_size, 1).to(self.device) # Duplicate CLS across the batch size
            if self.debug:   assert start_cls_emb.shape == (self.num_gumbel_samples, self.batch_size, 1, self.emb_size[model_name])
        else:  
            raise Exception(f"Unsupported tokenizer algorithm for tokenizer {tokenizer_name}")
        inputs_embeds = inputs_embeds.to(torch.float32)  # The forward method of models seems to need this. 
        attention_mask =  self.models[model_name]._prepare_attention_mask_for_generation(
                torch.concat([pre_ids, pp_ids[:,1:] ], dim=1),  # Remove the bos pad token from pp_ids + concat 
                self.tokenizers[model_name].pad_token_id, self.tokenizers[model_name].eos_token_id
        )        
        self.debug_prepare_model_inputs(pp_ids, weighted_emb, model_name, inputs_embeds, attention_mask)
        self.log_weighted_emb_postprocessing(model_name,  weighted_emb_before_change_len, inputs_embeds)
        return {'inputs_embeds':inputs_embeds, 'attention_mask': attention_mask}
    
    def _get_vm_logits_from_inputs_embeds(self, inputs):
        """Feed embeddings through the victim model and get logits"""
        inputs_embeds,attention_mask = inputs['inputs_embeds'], inputs['attention_mask']
        inputs_rep = inputs_embeds.reshape((self.num_gumbel_samples * self.batch_size, inputs_embeds.shape[2],inputs_embeds.shape[3]))
        attention_rep = attention_mask.repeat((self.num_gumbel_samples, 1))
        vm_logits = self.vm_model(inputs_embeds=inputs_rep, attention_mask=attention_rep).logits.squeeze()
        vm_logits = vm_logits.reshape((self.num_gumbel_samples, self.batch_size, self.dataset_num_labels))
        self.debug_get_vm_logits_from_inputs_embeds(inputs_embeds, attention_mask, vm_logits, inputs_rep, attention_rep)
        return vm_logits
    
    def _get_ld_logits_from_inputs_embeds(self, inputs):
        """Feed embeddings through the victim model and get logits"""
        inputs_embeds,attention_mask = inputs['inputs_embeds'], inputs['attention_mask']
        inputs_rep = inputs_embeds.reshape((self.num_gumbel_samples * self.batch_size, inputs_embeds.shape[2],inputs_embeds.shape[3]))
        attention_rep = attention_mask.repeat((self.num_gumbel_samples, 1))
        ld_logits = self.ld_model(inputs_embeds=inputs_rep, attention_mask=attention_rep).logits.squeeze()
        ld_logits = ld_logits.reshape((self.num_gumbel_samples, self.batch_size, self.ld_num_langs))
        self.debug_get_ld_logits_from_inputs_embeds(inputs_embeds, attention_mask, ld_logits, inputs_rep, attention_rep)
        return ld_logits

    def _get_sts_scores_and_diversity_score_from_inputs_embeds(self, orig_sts_embeddings, inputs): 
        # See the forward methods for 
        # https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py
        # and https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py
        # for details on how this code is structured
        inputs_embeds,attention_mask = inputs['inputs_embeds'],inputs['attention_mask']
        inputs_rep = inputs_embeds.reshape((self.num_gumbel_samples * self.batch_size, inputs_embeds.shape[2],inputs_embeds.shape[3]))
        attention_rep = attention_mask.repeat((self.num_gumbel_samples, 1))
        orig_rep = orig_sts_embeddings.repeat((self.num_gumbel_samples, 1))
        token_embeddings = self.sts_base_model.forward(inputs_embeds=inputs_rep, attention_mask=attention_rep)['last_hidden_state']
        features = dict(inputs_embeds=inputs_rep, attention_mask=attention_rep, token_embeddings=token_embeddings)
        pp_sts_embedding = self.sts_pooling_layer.forward(features)['sentence_embedding']
        ### STS SCORES
        sts_scores = pytorch_cos_sim(orig_rep, pp_sts_embedding).diagonal()        # training case
        sts_scores = sts_scores.reshape((self.num_gumbel_samples, self.batch_size))
        
        ### Intra-batch diversity penalty
        # Reshape the embeddings so that all Gumbel samples for the same example are grouped together, take mean
        pp_mean_embeddings_across_gumbel = pp_sts_embedding.reshape((self.num_gumbel_samples, self.batch_size, -1)).mean(dim=0)
        # Calc the sim matrix for orig embeddings - use this as a reference for the "ideal" amount of diversity. 
        # Calc the sim matrix for generated pp embeddings, also
        sim_matrix_orig = pytorch_cos_sim(orig_sts_embeddings,              orig_sts_embeddings)
        sim_matrix_pp   = pytorch_cos_sim(pp_mean_embeddings_across_gumbel, pp_mean_embeddings_across_gumbel)
        
        upper_triangle_diff = sim_matrix_orig.triu(diagonal=1) - sim_matrix_pp.triu(diagonal=1)
        sq_diff =  torch.square(upper_triangle_diff)
        # take mean of only the non-zero upper triangle elements (to keep it independent of batch size)
        diversity_score = torch.mean(sq_diff[sq_diff != 0])
        self.debug_get_sts_scores_and_diversity_score_from_inputs_embeds(inputs_embeds, attention_mask, orig_sts_embeddings, inputs_rep, attention_rep, orig_rep,
                                                                     pp_sts_embedding, sts_scores, sim_matrix_orig, sim_matrix_pp, diversity_score)
        self.debug_log_diversity_component_histograms(sim_matrix_orig, sim_matrix_pp, sq_diff)
        return sts_scores,diversity_score

    def _get_kl_div_from_inputs_embeds(self, orig_ids, orig_attention_mask, gumbel_probs, inputs): 
        """orig_ids and orig_attention_mask are ids+attention mask from the orig ids from pp_tokenizer. 
        inputs are a dict of the processed inputs_embeds and the corresponding attention mask. 
        """
        inputs_embeds,attention_mask_inputs = inputs['inputs_embeds'], inputs['attention_mask']
        pp_probs_all = gumbel_probs
        pp_logprobs_all = torch.log(pp_probs_all)
        pp_logprobs_all = torch.nan_to_num(pp_logprobs_all, nan=None, posinf=None, neginf=-20)  # -inf screws things up
        pp_logprobs_all = pp_logprobs_all.clip(min=-20)
        l1 = []
        for b in range(self.num_gumbel_samples):
            kl_divs_l = []
            for i in range(self.batch_len_pp_ids_pp_tknzr - 1):
                decoder_inputs_embeds = inputs_embeds[b, :, 0:(i+1), :]
                input_d = {'input_ids':orig_ids, 'attention_mask':orig_attention_mask, 'decoder_inputs_embeds':decoder_inputs_embeds}
                ref_outputs = self.models['ref'](**input_d)
                ref_logprobs = ref_outputs.logits[:, i, :].log_softmax(1)
                ref_logprobs = torch.nan_to_num(ref_logprobs, nan=None, posinf=None, neginf=-20)  # -inf screws things up
                ref_logprobs = ref_logprobs.clip(min=-20)
                pp_probs    = pp_probs_all   [b, :, i, :]
                pp_logprobs = pp_logprobs_all[b, :, i, :]
                # KL div:  p_pp dot (log(p_pp) -  log(p_ref))
                kl_div = torch.mm(pp_probs, (pp_logprobs - ref_logprobs).t()).diagonal()
                kl_divs_l.append(kl_div)
                self.debug_kl_div_inner_loop(i, decoder_inputs_embeds, ref_outputs, ref_logprobs, pp_probs, pp_logprobs, kl_div)
            kl_divs = torch.stack(kl_divs_l, 1)
            kl_divs = kl_divs * attention_mask_inputs[:, 1:]  # account for padding 
            kl_divs_normalised = kl_divs.sum(1) / attention_mask_inputs[:, 1:].sum(1)  # normalise for length of generated sequence
            l1.append(kl_divs_normalised)
            self.debug_kl_div_outer_loop(kl_divs, kl_divs_normalised)
        kl_divs_stacked = torch.stack(l1, 0)
        self.debug_get_kl_div_from_inputs_embeds(inputs_embeds, attention_mask_inputs, gumbel_probs, kl_divs_stacked, pp_probs_all, pp_logprobs_all)
        return kl_divs_stacked

    def _loss_fn_example(self, vm_scores, sts_scores, ld_scores, **kwargs): 
        if type(vm_scores)  == float: vm_scores  = torch.tensor(vm_scores)
        if type(sts_scores) == float: sts_scores = torch.tensor(sts_scores)
        if type(ld_scores)  == float: ld_scores  = torch.tensor(ld_scores)
        # if type(nli_scores) == float: nli_scores = torch.tensor(nli_scores)
        vm_component  = self.args.coef_vm  * vm_scores
        sts_component = self.args.coef_sts * sts_scores
        ld_component  = self.args.coef_ld  * ld_scores

        if 'training' in kwargs and kwargs['training'] == True: 
            self.log_dict({'vm_component_batch_mean': torch.mean(vm_component).item(), 
                    'sts_component_batch_mean':torch.mean(sts_component).item(),
                    'ld_component_batch_mean': torch.mean(ld_component).item()}, on_step=True)
        return -(vm_component + sts_component - ld_component)

    def loss_fn(self, vm_scores, sts_scores, ld_scores, kl_divs, diversity_score, **kwargs): 
        """batch one"""
        # Loss clipping
        # we take in a tensor. 
        # for each element in tensor, if we are in a "bad range" (i.e. the where condition is met) then we get a True in that element, else False. 
        # wherever we have True, we keep the original value. Wherever we have False, we set to 0. 
        # e.g. Sts scores are [0.9, 0.2, 0.5, 0.1], and eval_sts_threhsold is 0.4. 
        # this maps to [False, True, False, True] and [0, 0.2, 0, 0.1]
        # this is multipled by coef (e.g. 2) to get [0, 0.4, 0, 0.1]
        # Then this is added to the loss fn in -(vm_component + sts_component + ld_component)
        vm_scores_clipped  = torch.where(vm_scores  < self.eval_vm_threshold ,              vm_scores,  torch.zeros_like(vm_scores))        
        sts_scores_clipped = torch.where(sts_scores < self.args.eval_sts_threshold,          sts_scores, torch.zeros_like(sts_scores))
        ld_scores_clipped  = torch.where(ld_scores  > self.args.eval_ld_threshold,          ld_scores,  torch.zeros_like(ld_scores))
        kl_divs_clipped    = torch.where(kl_divs    > self.args.eval_kl_threshold,           kl_divs,    torch.zeros_like(kl_divs))
        # Penalties. Lower is better. In range [0, inf) - unbounded positively. 
        kl_penalty =  self.args.coef_kl * torch.mean(kl_divs_clipped)   # lower (towards 0) better 
        diversity_penalty = self.args.coef_diversity * diversity_score   # lower (towards 0) better 
        # Final loss fn
        loss_examples = self._loss_fn_example(vm_scores=vm_scores_clipped, sts_scores=sts_scores_clipped, ld_scores=ld_scores_clipped, training=True) 
        loss_examples_mean = torch.mean(loss_examples)
        loss_batch = loss_examples_mean + kl_penalty + diversity_penalty  # want as negative as possible
        self.debug_log_scores_histograms(vm_scores_clipped, sts_scores_clipped, ld_scores_clipped, kl_divs_clipped)
        self.log_batch_losses(kl_penalty, diversity_penalty, loss_examples_mean, loss_batch)
        return {'loss_batch':loss_batch}

    def training_step(self, batch, batch_idx):
        """complete training loop"""
        if not self.pp_model.training: self.pp_model.training() 
        for k,model in self.models.items(): 
            if k != "pp" and model.training: model.eval() # lightning seems to automatically set models out of eval mode sometimes
        forward_d = self(batch)
        loss_d = self.loss_fn(**forward_d)        
        return_d = {'loss': loss_d['loss_batch'], **{k: v for k, v in batch.items() if k not in self.orig_cols_to_ignore}}
        return_d = self.training_step_add_forward_d(forward_d, return_d)
        return return_d  # must include the key 'loss' 
  
    def eval_step(self, batch, batch_idx, eval_setting): 
        if self.vm_model.training: self.vm_model.eval()
        if self.pp_model.training: self.pp_model.eval()
        forward_d = self.forward_eval(batch, eval_setting)
        return {**{k: v for k, v in batch.items() if k not in self.orig_cols_to_ignore},  **forward_d}
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Complete validation loop"""
        results = self.eval_step(batch, batch_idx, eval_setting='val')
        return results
        
    def test_step(self, batch, batch_idx):
        """complete testing loop"""
        for eval_setting in self.gen_config_eval_d.keys(): 
            if eval_setting != 'val': 
                self.test_results_d[eval_setting].append(self.eval_step(batch, batch_idx, eval_setting))
        return self.test_results_d

    def predict_step(self, batch, batch_idx):
        results = self.eval_step(batch, batch_idx)
        return results

    def is_label_flip(self, labels, vm_predclass): return ((vm_predclass != labels) * 1)

    def is_valid_pp(self, sts_scores, ld_scores, kl_divs):  
        return (sts_scores > self.args.eval_sts_threshold  and ld_scores < self.args.eval_ld_threshold and kl_divs < self.args.eval_kl_threshold)

    def is_adv_example(self, label_flip, is_valid): return label_flip and is_valid

    def _convert_end_of_epoch_metrics_to_pandas_df(self, outputs, split, gen_setting=None): 
        """Takes the output metrics, converts them to a pandas dataframe, 
            and appends it to the appropiate entry in self.df_l"""
        # Outputs -> dataframe skeleton 
        df = pd.DataFrame(outputs).apply(pd.Series.explode).reset_index(drop=True)  # list[dict] -> dataframe. one row per orig
        df = df.applymap(lambda x: x.item() if is_0d_tensor(x) else x) #  one-element tensors -> float/int scalars
       
        # Eval-specific preprocessing.
        if split != "training":
            # Eval has multiple pp per orig: train has one
            df = unpack_nested_lists_in_df(df, scalar_cols=df.select_dtypes(np.number).columns.tolist())  # one row per pp. scalar_cols fn will not pick up tensors
            df = df.rename({'pp_text_nested': 'pp_text'}, axis=1)
             # Add vm scores (already included in training)
            vm_scores_d = get_vm_scores_from_vm_logits(
                labels=torch.tensor(df['label'].values, device=self.device),
                orig_truelabel_probs=torch.tensor(df['orig_truelabel_probs'].values, device=self.device),
                vm_logits=torch.stack(df['vm_logits_nested'].values.tolist())
            )
            for k, v in vm_scores_d.items(): vm_scores_d[k] = v.cpu()
            df = pd.concat([df, pd.DataFrame(vm_scores_d)], axis=1)
            df = df.drop(columns='vm_logits_nested')

        # Add metric columns  
        def get_mean(c): return np.mean(df[f'{c}_mean'])   # we drop incomplete batches so they should all be means over the same number of elements
        s = 'train' if split == "training" else split
        if split == 'training': 
            df['global_step'] =  self.global_step 
            # Metric dict for wandb
            d = {
                f'vm_score_{s}_mean': get_mean('vm_scores'),
                f'sts_score_{s}_mean': get_mean('sts_scores'), 
                f'ld_score_{s}_mean': get_mean('ld_scores'), 
                f'kl_div_{s}_mean': get_mean('kl_divs'),
            }
        if split != 'training': 
            # Calc paraphrase index and evaluation metric
            df['loss_example'] =  df.apply(lambda x: self._loss_fn_example(**x).item(), axis=1)
            df['pp_idx'] = df.groupby(['idx']).cumcount()
            df['label_flip'] = df.apply(lambda x: self.is_label_flip(labels=x.label,  vm_predclass=x.vm_predclass), axis=1)
            df['is_valid_pp'] = df.apply(lambda x: self.is_valid_pp(sts_scores=x.sts_scores, ld_scores=x.ld_scores, kl_divs=x.kl_divs) * 1, axis=1) 
            df['is_adv_example'] = df.apply(lambda x: self.is_adv_example(label_flip=x.label_flip, is_valid=x.is_valid_pp) * 1, axis=1)
            any_label_flip = (df.groupby('idx')['label_flip'].mean() > 0)*1
            any_adv_example = (df.groupby('idx')['is_adv_example'].mean() > 0)*1
            df['quality_score'] = df.apply(lambda x:  self.get_linguistic_quality_score( 
                            sts_score=x.sts_scores, ld_score=x.ld_scores, kl_div=x.kl_divs), axis=1) 
            df['adv_score'] = df.apply(lambda x:  self.get_adv_quality_score( 
                            vm_score=x.vm_scores, sts_score=x.sts_scores, ld_score=x.ld_scores, kl_div=x.kl_divs), axis=1) 
            if   split == 'validation':                df['global_step'] = self.global_step
            elif split == 'test': 
                if getattr(self, 'best_global_step', None) is not None:  df['global_step'] = self.best_global_step
                else:                                                     df['global_step'] = 0
            g = f'_{gen_setting}' if gen_setting else ""
            d = dict()
            if split == 'validation': # don't need these for test
                d = {
                    f'loss_{s}{g}_mean': df.loss_example.mean(), 
                    f'vm_score_{s}{g}_mean': df.vm_scores.mean(), 
                    f'sts_score_{s}{g}_mean': df.sts_scores.mean(), 
                    f'ld_score_{s}{g}_mean': df.ld_scores.mean(), 
                    f'kl_div_{s}{g}_mean': df.kl_divs.mean(),
                    f'adv_score_{s}{g}_mean': df.adv_score.mean()
                }
                d[f'any_adv_example_proportion_{s}{g}'] = any_adv_example.mean()
                d[f'ref_logp_{s}{g}_mean'] = df['ref_logp'].mean()
                d[f'pp_logp_{s}{g}_mean']   = df['pp_logp'].mean()
                d[f'label_flip_{s}{g}_mean']    =  df['label_flip'].values.mean()
                d[f'is_valid_pp_{s}{g}_mean']   =  df['is_valid_pp'].values.mean()
                d[f'is_adv_example_{s}{g}_mean']=  df['is_adv_example'].values.mean()
            u = f"_untrained" if self.untrained_run else ""
            d[f'{s}_attack_success_rate{u}{g}'] = any_label_flip.mean()
            self.asr_d[f'{s}_attack_success_rate{u}{g}'] = d[f'{s}_attack_success_rate{u}{g}']
        # Log + return
        self.log_dict(d, sync_dist=True)
        self.df_l[split].append(df)
        return d

    def training_epoch_end(self, outputs: List[dict]):
        # We need to summarise metrics across all gumbel samples
        gumbel_keys = [k for k,v in outputs[0].items() if type(v) is not list and len(v.shape) == 2]
        results_l = list() 
        for output in outputs:
            results_d = dict()
            for k in gumbel_keys: 
                if k =='vm_predclass':  # We just look at the fraction of label flips
                    vm_predclass_same_labels = output['label'].repeat((self.num_gumbel_samples, 1)).cpu() != output['vm_predclass']
                    assert vm_predclass_same_labels.shape == (self.num_gumbel_samples, output['vm_predclass'].shape[1])
                    label_flip_frac = torch.sum(vm_predclass_same_labels, axis=0) / self.num_gumbel_samples
                    results_d['label_flip_frac'] = label_flip_frac
                else: 
                    # all other metrics - just get mean and (min, 25%, 50%, 75%, max)
                    results_d[f'{k}_mean'] = output[k].mean(axis=0)
                    summary_stats = output[k].quantile(torch.tensor([0, 0.25, 0.5, 0.75, 1]), dim=0, interpolation='linear')
                    results_d[f'{k}_quantiles']= summary_stats.t().tolist()
            results_l.append(results_d)                    
        new_outputs = [{**{k: v for k, v in d_orig.items() if k not in gumbel_keys}, **d_new} for d_orig,d_new in zip(outputs, results_l)]
        if self.debug: 
            assert(len(results_l) == len(outputs))
            assert len(new_outputs) == len(outputs)
        self._convert_end_of_epoch_metrics_to_pandas_df(new_outputs, split='training')

    def validation_epoch_end(self, outputs):
        self._convert_end_of_epoch_metrics_to_pandas_df(outputs, split='validation')

    def test_epoch_end(self, results_d): 
        # results_d: dict with keys as generation settings, values as output from eval_step (like validation_epoch_end)
        from src.eval_metrics import set_up_all_scorers
        self._move_models_to_device('cpu')  # clear GPU space 
        # Set up evaluation metrics
        self.scorer_d =  set_up_all_scorers(device=self.device)
        for k, outputs in self.test_results_d.items(): 
            self.test_fname = f'{self.path_run}/test{"_untrained" if self.untrained_run else ""}_{k}.csv'
            self._convert_end_of_epoch_metrics_to_pandas_df(outputs, split='test', gen_setting=k)
            self.save_results_df(split='test', untrained_run=self.untrained_run)#, gen_setting=k)
            self.calc_and_log_eval_metrics(untrained_run=self.untrained_run, gen_setting=k)

        # clean up
        del self.scorer_d
        torch.cuda.empty_cache()
        gc.collect()
        self._move_models_to_device(self.device)
        # reset testing datastores
        self.test_results_d = dict()
        for eval_setting in self.gen_config_eval_d.keys():
            if eval_setting != 'val': self.test_results_d[eval_setting] = []
 
    def _move_models_to_device(self, device): 
        for k,model in self.models.items():
            for param in model.parameters():
                param.data = param.data.to(device)
        for k in ['vm', 'sts', 'ld']: 
            self.__setattr__(f"token_mapping['{k}']",self.__getattr__(f"token_mapping['{k}']").to(device) )
        torch.cuda.empty_cache()
        gc.collect()

    def save_results_df(self, split, untrained_run=False): 
        df_all = pd.concat(self.df_l[split])
        if self.args.run_mode == 'sweep':
            fname = f'{self.path_run}/orig_validation.csv'
        else: 
            fname = f'{self.path_run}/orig_{"train" if split == "training" else split}.csv'
        df_dataset = pd.read_csv(fname)
        cols_to_remove = list(set(df_all.columns.to_list()).intersection(set(df_dataset.columns.to_list()))); cols_to_remove.remove('idx')
        cols_to_remove += self.orig_cols_to_ignore        
        df_dataset = df_dataset.drop(columns=cols_to_remove, errors='ignore')
        df_final = pd.merge(left=df_all, right=df_dataset, on='idx', how='left')
        fname =  self.test_fname if split =="test" else f'{self.path_run}/{split}{"_untrained" if untrained_run else ""}.csv'
        df_final.to_csv(fname, index=False)
        if untrained_run or split=='test': self.df_l[split] = []  # reset when doing the epoch 0/untrained ones

    def peft_train(self): 
        """Get the model ready for training"""
        if self.args.use_peft:  self.pp_model.unmerge_adapter() 
        self.pp_model.train()

    def peft_eval(self):
        """Get the model ready for evaluation"""
        if self.args.use_peft:  self.pp_model.merge_adapter()  
        self.pp_model.eval()

    def on_train_start(self):           
        self.peft_train()

    def on_validation_start(self):      self.peft_eval()

    def on_validation_end(self):        self.peft_train()

    def on_test_start(self):            self.peft_eval()

    def on_train_end(self): 
        # runs after the whole training and validation procedure
        # test is in self.test_epoch_end
        self.save_results_df(split='training')
        self.save_results_df(split='validation')

    def get_adv_quality_score(self, vm_score, sts_score, ld_score, kl_div):
        """Score that combines the linguistic quality score with the vm score. """
        quality = self.get_linguistic_quality_score(sts_score=sts_score, ld_score=ld_score, kl_div=kl_div)
        vm_score = min(vm_score, self.eval_vm_threshold)  # doesn't matter if we go over the threshold
        return quality + 7 *  vm_score


    def get_linguistic_quality_score(self, sts_score, ld_score, kl_div):
        """Score to determine 'linguistic quality' of a candidate. """ 
        ld_score  = max(ld_score,  0)  # Negative LD score doesn't matter
        kl_div    = abs(kl_div)   # some v negative values aren't good
        return   0.75 * sts_score - ld_score - 0.6 * kl_div 

    def select_test_eval_examples(self, test_fname): 
        """selects out label flips, selects "champion" from each example for eval metrics to be applied. """
        df_label_flips  = pd.read_csv(test_fname).query('label_flip==1')
        if len(df_label_flips) == 0: 
            print(f'No label flips detected in {test_fname}. Skipping metrics')
            return
        def select_best_row(group):
            """We just use quality score among label flips"""
            return group.loc[group['quality_score'].idxmax()]
        
        df_label_flips = df_label_flips.groupby('idx').apply(select_best_row).reset_index(drop=True)
        df_label_flips = df_label_flips.sort_values('quality_score', ascending=False).groupby('idx').head(1)
        if type(df_label_flips) == pd.Series: df_label_flips = df_label_flips.to_frame().T
        df_label_flips = df_label_flips.rename(columns={'pp':'pp_text', 'orig': 'text'})
        df_label_flips = df_label_flips.rename(columns={'sentence': 'text'})
        return df_label_flips

    def log_eval_metrics_to_wandb(self, df_with_metrics, untrained_run, gen_setting): 
        u = "_untrained"      if untrained_run else ""
        g = f"_{gen_setting}" if gen_setting   else ""
        d = {
            f"test_flu{u}{g}_avg":              df_with_metrics['flu'].mean(),
            f"test_flu{u}{g}_median":           df_with_metrics['flu'].median(),
            f"test_langscore{u}{g}_avg":        df_with_metrics['langscore'].mean(),
            f"test_langscore{u}{g}_median":     df_with_metrics['langscore'].median(),
            f"test_sim{u}{g}_avg":              df_with_metrics['sim'].mean(),
            f"test_sim{u}{g}_median":           df_with_metrics['sim'].median(),
        }
        vsr_d = get_vsr_d(df_with_metrics)
        vsr_d_renamed = dict()
        for k, v in vsr_d.items(): 
            vsr_d_renamed[f"{k}{u}{g}"] = v
        self.log_dict(d)
        self.log_dict(vsr_d_renamed)

    def calc_and_log_eval_metrics(self, untrained_run=False, gen_setting=None):
        from src.eval_metrics import score_df
        df_eval_examples = self.select_test_eval_examples(self.test_fname) 
        if df_eval_examples is None: return 
        df_with_metrics = score_df(df_eval_examples, self.scorer_d, cname_orig='text', cname_pp='pp_text', device=self.device)
        
        self.log_eval_metrics_to_wandb(df_with_metrics, untrained_run=untrained_run, gen_setting=gen_setting)
        fname = self.test_fname[:-4] + "_evaluation.csv"
        df_with_metrics.to_csv(fname, index=False)
        

    def configure_optimizers(self):
        """optimizers"""
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        if self.args.optimizer_type == 'AdaFactor': 
            self.learning_rate = 3e-4 if self.args.learning_rate is None else self.args.learning_rate  # a good default for adafactor
            optimizer = Adafactor(parameters, scale_parameter=False, relative_step=False, warmup_init=False, lr=self.learning_rate)
        elif self.args.optimizer_type == 'AdamW': 
            # seems to be a good default for adamw too, by coincidence same as adafactor 
            self.learning_rate = 3e-4 if self.args.learning_rate is None else self.args.learning_rate  
            optimizer = AdamW(parameters, lr=self.learning_rate, weight_decay=self.args.weight_decay)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):     
        parser = parent_parser.add_argument_group("T5ForTextClassification")
        parser.add_argument("--num_gumbel_samples", type=int, help="number of gumbel samples to take for each input")
        parser.add_argument("--gumbel_tau", type=float, help="Tau parameter for gumbel-softmax sampling.")
        ## LOSS FN 
        parser.add_argument("--coef_vm",  type=float,  help="coefficient for the vm scores in the exampleloss (should be +ve)")
        parser.add_argument("--coef_sts", type=float,  help="coefficient for the sts scores in the example loss (should be +ve)")
        parser.add_argument("--coef_ld", type=float,  help="coefficient for the language detection scores in the example loss (should be +ve)") # TODO
        parser.add_argument("--coef_kl",  type=float,  help="coefficient for the kl divergence in the batch loss (should be +ve)")
        parser.add_argument("--coef_diversity",  type=float, help="coefficient for the diversity component in the  batch loss (should be +ve)")

        ## EVAL THRESHOLDS
        parser.add_argument("--eval_sts_threshold",  type=float, help="minimum sts score for a valid adv example during eval")
        parser.add_argument("--eval_ld_threshold",  type=float, help="minimum ld score for a valid adv example during eval")  # TODO
        parser.add_argument("--eval_kl_threshold",   type=float, help="maximum kl div for a valid adv example during eval")
        
        ## GEN PARAMETERS
        parser.add_argument("--eval_condition",            type=str,  choices=['standard', 'dev', 'ablation'],
            help="Evaluation preset conditions for generation. Will set all the eval generation paramters. See utils.set_eval_gen_settings_from_eval_condition")
        return parent_parser


