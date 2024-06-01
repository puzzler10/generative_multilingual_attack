from pprint import pprint
import ssl
from src.models import get_vm_tokenizer_and_model 
import src.parsing_fns 
import wandb
import transformers
import src.eval_metrics
import datetime 
import pandas as pd 
import time
from src.training_fns import load_preprocessed_dataset

import warnings
warnings.filterwarnings("ignore", message="FutureWarning: The frame.append method is deprecated") 
from src.eval_metrics import set_up_all_scorers,score_df

from textattack import Attack, AttackArgs, Attacker
from textattack.search_methods import GreedySearch, GreedyWordSwapWIR
from textattack.goal_functions import UntargetedClassification
from textattack.transformations import  WordSwapMaskedLM,CompositeTransformation,WordInsertionMaskedLM,WordMergeMaskedLM, WordSwapMaskedLM
from textattack.constraints.pre_transformation import RepeatModification
from textattack.constraints.semantics.sentence_encoders.universal_sentence_encoder import MultilingualUniversalSentenceEncoder
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.metrics.attack_metrics.attack_success_rate import AttackSuccessRate
from textattack.metrics.attack_metrics.words_perturbed import WordsPerturbed
from textattack.metrics.attack_metrics.attack_queries import AttackQueries
from textattack.datasets import HuggingFaceDataset
from textattack.attack_recipes import AttackRecipe

# Commented-out imports
from textattack.models.wrappers import HuggingFaceModelWrapper


def display_adv_example(df): 
    from IPython.core.display import display, HTML
    pd.options.display.max_colwidth = 480 # increase column width so we can actually read the examples
    display(df[['original_text', 'perturbed_text']])


class BERTAttackMultilingual(AttackRecipe):
    """Adapting BERTAttack to the multilingual setting. No language detection components. """
    @staticmethod
    def build(model_wrapper):
        transformation = WordSwapMaskedLM(
            method="bert-attack",
            max_candidates=6,
            masked_language_model="bert-base-multilingual-cased")
        constraints = [RepeatModification()] #  StopwordModification()
        constraints.append(MaxWordsPerturbed(max_percent=0.4))
        use_constraint = MultilingualUniversalSentenceEncoder(
            threshold=0.2,
            metric="cosine",
            compare_against_original=True,
            window_size=None,
        )
        constraints.append(use_constraint)
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedyWordSwapWIR(wir_method="unk")
        return Attack(goal_function, constraints, transformation, search_method)


class CLAREMultilingual(AttackRecipe):
    @staticmethod
    def build(model_wrapper):
        shared_masked_lm = transformers.AutoModelForCausalLM.from_pretrained("bert-base-multilingual-cased")
        shared_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
        transformation = CompositeTransformation(
            [
                WordSwapMaskedLM(
                    method="bae",
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=5e-4,
                ),
                WordInsertionMaskedLM(
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=0.0,
                ),
                WordMergeMaskedLM(
                    masked_language_model=shared_masked_lm,
                    tokenizer=shared_tokenizer,
                    max_candidates=50,
                    min_confidence=5e-3,
                ),
            ]
        )
        constraints = [RepeatModification()] # StopwordModification()
        use_constraint = MultilingualUniversalSentenceEncoder(
            threshold=0.7,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedySearch()
        return Attack(goal_function, constraints, transformation, search_method)



class BAEMultilingual(AttackRecipe):
    """BAE-R, multilingual version."""
    def build(model_wrapper):
        transformation = WordSwapMaskedLM(method="bae", max_candidates=50, min_confidence=0.0, masked_language_model="bert-base-multilingual-cased")
        constraints = [RepeatModification()] # , StopwordModification()
        use_constraint = MultilingualUniversalSentenceEncoder(
            threshold=0.936338023,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedyWordSwapWIR(wir_method="delete")
        return Attack(goal_function, constraints, transformation, search_method)



def main(args):
    # update this with your entity and project if using wandb
    wandb.init(project="your_project_here", entity="your_entity_here", mode=args.wandb_mode)
    wandb.config.update(args)
    path_baselines = "./results/"
    datetime_now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"{path_baselines}{datetime_now}_{args.dataset_name}_{args.lang}.csv"
    print(filename)
    device = 'cuda:0' if args.accelerator == "gpu" else "cpu"

    # Load dataset 
    # we will just use the text components, rest doesn't matter
    fname = f"data/adversary_datasets/DS--{args.dataset_name}--LANG--{args.lang}--PP--mt5_base_paraphrase--STS--sentence_transformers_paraphrase_multilingual_minilm_l12_v2--LD--dunnbc22_distilbert_base_multilingual_cased_language_detection.pkl"
    dataset = load_preprocessed_dataset(args, fname=fname).dsd['test']
    if args.n_examples != -1:    dataset = dataset.select(list(range(args.n_examples)))
    dataset_info_df = dataset.select_columns(['idx', 'lang', 'label']).to_pandas()
    hf_dataset = HuggingFaceDataset(dataset, dataset_columns=(['text'], 'label')) 

    # load victim model 
    vm_tokenizer, vm_model   = get_vm_tokenizer_and_model(args)
    vm_model.to(device)
    # check on GPU here 
    model_wrapper = HuggingFaceModelWrapper(vm_model, vm_tokenizer)


    if   args.attack_name == 'bertattack_multilingual':  attack_recipe = BERTAttackMultilingual
    elif args.attack_name == 'clare_multilingual':       attack_recipe = CLAREMultilingual
    elif args.attack_name == 'bae_multilingual':       attack_recipe = BAEMultilingual
    # We filter the dataset down to only the number of examples prior, so put -1
    attack_args = AttackArgs(num_examples=-1, shuffle=False, query_budget=args.query_budget,
                             log_to_csv=filename, csv_coloring_style='plain', disable_stdout=False)
    attack = attack_recipe.build(model_wrapper)
    attacker = Attacker(attack, hf_dataset, attack_args)

    start_time = time.time()
    attack_results = attacker.attack_dataset()
    end_time = time.time()
    time_taken = end_time - start_time


    attack_result_metrics = {
        **AttackSuccessRate().calculate(attack_results), 
        **WordsPerturbed().calculate(attack_results),
        **AttackQueries().calculate(attack_results),
        "attack_time": time_taken
    }
    wandb.log(attack_result_metrics)
    # read in the results
    df_attack = pd.read_csv(filename)

    df_all_attacks = pd.concat([dataset_info_df, df_attack], axis=1)
    df_all_attacks['flip'] = [1 if x == 'Successful' else 0 for x in df_all_attacks['result_type']]
    scorer_d = set_up_all_scorers(device)
    df_with_metrics = score_df(df_all_attacks, scorer_d,  cname_orig='original_text', cname_pp='perturbed_text')
    df_successes = df_with_metrics.query('flip==1')
    d = {
        f"flu_avg":              df_successes['flu'].mean(),
        f"flu_median":           df_successes['flu'].median(),
        f"langscore_avg":        df_successes['langscore'].mean(),
        f"langscore_median":     df_successes['langscore'].median(),
        f"sim_avg":              df_successes['sim'].mean(),
        f"sim_median":           df_successes['sim'].median(),
    }
    d['attack_success_rate'] = len(df_successes) / len(df_all_attacks)
    df_with_metrics.rename(columns={'flip': 'label_flip'}, inplace=True)
    info_d = {"attack_name": args.attack_name, "dataset_name": args.dataset_name, "lang": args.lang, "n_examples": args.n_examples,
               "run_mode": args.run_mode, "max_length_orig":args.max_length_orig}
    from src.eval_metrics import get_vsr_d
    vsr_d = get_vsr_d(df_with_metrics)   
    df_with_metrics.to_csv(f'{filename[:-4]}_processed.csv')
    result_d = {**info_d, **d, **vsr_d}
    wandb.log(result_d)

if __name__ == "__main__":
    args = src.parsing_fns.get_args_run_baselines()
    pprint(vars(args))
    main(args)




