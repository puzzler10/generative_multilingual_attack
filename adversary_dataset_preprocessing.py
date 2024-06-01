from argparse import  Namespace,ArgumentParser
from multiprocessing import cpu_count
import os
from src.models import get_pp_tokenizer_and_model, get_vm_tokenizer_and_model, get_sts_model, get_ld_tokenizer_and_model, load_reconstructed_mt5_tokenizer_and_model
from src.dataset_prep import AdversaryDataset
from src.training_fns import choose_gpu, get_dset_fname_from_args
import pickle
from src.training_fns import get_device_string
from datasets import DatasetDict, concatenate_datasets



def save_dataset_to_file(dataset, fname):
    # Create directory if it doesn't exist
    dir_name = os.path.dirname(fname)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # Pickle and save to file, overwriting if it exists
    with open(fname, 'wb') as f:
        pickle.dump(dataset, f)


def process_and_save_dataset(args, **kwargs):
    vm_tokenizer,   vm_model   = get_vm_tokenizer_and_model(args)
    vm_model.to(get_device_string(args))
    dataset = AdversaryDataset(args, **kwargs,  vm_tokenizer=vm_tokenizer, vm_model=vm_model, get_dataloaders=False)
    # save dataset object to file as pickle
    if args.dataset_name == 'amazon_reviews_multi':
        # if one language, shuffle (seed=0), select 5k/1k/1k as the training set
        if args.lang != 'all':
            dataset.dsd['train']      = dataset.dsd['train'].shuffle(seed=1).select(range(5000))
            dataset.dsd['validation'] = dataset.dsd['validation'].shuffle(seed=1).select(range(1000))
            dataset.dsd['test']       = dataset.dsd['test'].shuffle(seed=1).select(range(1000))
        else: #if "all", select 1.25/0.25/0.25k from each language, compile to 5k/1k/1k
            dsd1 = dict(train=[], validation=[], test=[])
            for lang in ['en', 'de', 'es', 'fr']:
                dsd1['train'].append(      dataset.dsd['train'].filter(     lambda x: x['lang'] == lang).shuffle(seed=1).select(range(1250)))
                dsd1['validation'].append( dataset.dsd['validation'].filter(lambda x: x['lang'] == lang).shuffle(seed=1).select(range(250)))
                dsd1['test'].append(       dataset.dsd['test'].filter(      lambda x: x['lang'] == lang).shuffle(seed=1).select(range(250)))
            dataset.dsd = DatasetDict(train=concatenate_datasets(dsd1['train']), 
                                      validation=concatenate_datasets(dsd1['validation']),
                                      test=concatenate_datasets(dsd1['test']))
            dataset.dsd = dataset.dsd.shuffle(seed=2)
    fname = get_dset_fname_from_args(args)
    save_dataset_to_file(dataset, fname)   


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default=None, help='Dataset name to override')
    parser.add_argument('--lang', type=str, default=None, help='Language to override')
    parser.add_argument('--pp_name', type=str, default=None, help='Paraphrase model name to override')
    return parser.parse_args()


if __name__ == "__main__":
    # Set up prepocessing args (the src.models and data functions need it currently ) 
    # For interactive mode, manually set args
    args = Namespace(
        pp_name=None,
        vm_name=None,
        sts_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ld_name="DunnBC22/distilbert-base-multilingual-cased-language_detection",
        download_models=False,
        freeze_vm_model=True,
        accelerator='gpu',
        seed=0,
        run_mode="prod"
    )
    args.num_workers = min(16, cpu_count() - 1)
    args.min_length_orig = 0
    args.max_length_orig = 32  # that's what paraphraser was trained on 
    args.long_example_behaviour = "remove"
    args.bucket_by_length = True
    args.disable_hf_caching = False
    args.devices=[choose_gpu()]
    args.n_shards=-1

    # used when loading dataloaders
    args.shuffle_train = False
    args.shuffle_buckets=True
    args.batch_size = 4
    args.batch_size_eval = 4

    # Override default args with command-line args if provided
    cmd_args = parse_args()
    if cmd_args.dataset_name:
        args.dataset_name = cmd_args.dataset_name
    if cmd_args.lang:
        args.lang = cmd_args.lang
    if cmd_args.pp_name:
        args.pp_name = cmd_args.pp_name    

    if args.lang:   langs = [args.lang]
    else:           langs = ['all','ar', 'en', 'de', 'es', 'fr']

    if args.pp_name:   pp_names = [args.pp_name]
    else:              pp_names = ["mt5_small_paraphrase", "mt5_base_paraphrase"]
    if args.dataset_name:   datasets = [args.dataset_name]
    else:                   datasets = ["amazon_reviews_multi", "tweet_sentiment_multilingual"]
    

    # Load constant models 
    model_path = "path_to_model"  # you gotta update this
    ld_tokenizer,   ld_model    = get_ld_tokenizer_and_model(args)
    sts_model    = get_sts_model(args)
    ld_model.to(get_device_string(args))
    sts_model.to(get_device_string(args))
    pp_names = ["mt5_small_paraphrase", "mt5_base_paraphrase"]
    for pp_name in pp_names:
        # Load pp model
        args.pp_name = f"{model_path}/{pp_name}.ckpt"    
        if "reconstructed" in args.pp_name:  
            path_mt5 = f"{model_path}/mt5_reconstructed"
            pp_tokenizer,   pp_model  = load_reconstructed_mt5_tokenizer_and_model(args, path_mt5)
        else:                              
            pp_tokenizer,   pp_model  = get_pp_tokenizer_and_model(args)
        # Cycle through languages
        pp_model.to(get_device_string(args))
        for dataset in datasets:
            args.dataset_name = dataset
            for lang in langs: 
                args.lang=lang
                print("PP_MODEL", pp_name)
                print("LANG", lang)
                print("DATASET", args.dataset_name)
                args.vm_name = f"{model_path}/{args.dataset_name}_{args.lang}.ckpt"
                if args.dataset_name == 'amazon_reviews_multi': 
                    if lang != 'ar': 
                        process_and_save_dataset(args, pp_tokenizer=pp_tokenizer, ld_tokenizer=ld_tokenizer,
                                            sts_model=sts_model, ld_model=ld_model)
                else: 
                    process_and_save_dataset(args, pp_tokenizer=pp_tokenizer, ld_tokenizer=ld_tokenizer,
                                sts_model=sts_model, ld_model=ld_model)
