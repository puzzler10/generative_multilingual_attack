from pytorch_lightning import Trainer
from argparse import ArgumentParser
from distutils.util import strtobool
from multiprocessing import cpu_count
from random import randint
from src.adversary import MultilingualWhiteboxAdversary
from src.victim_model import VictimModel
from src.training_fns import choose_gpu, delete_lockfiles
from src.utils import get_least_occupied_gpu

def add_common_args(parser): 
    common_args = parser.add_argument_group('Common arguments for diff projects')
    ## Data
    common_args.add_argument("--dataset_name", type=str, 
        choices=['tweet_sentiment_multilingual','amazon_reviews_multi'], 
        help="The name of the dataset to use.")
    common_args.add_argument("--lang", type=str, 
                             choices=['ar', 'en', 'fr', 'de', 'es', 'all'], help="The language to use.")
    common_args.add_argument('--max_length_orig', type=int, 
        help="Set to a value to remove any examples with more tokens that that from the dataset.")
    common_args.add_argument('--min_length_orig', type=int,
        help="Minimum number of tokens in original.")
    common_args.add_argument('--shuffle_train', type=lambda x: bool(strtobool(x)),
        help="Shuffle the training set during training. Cannot be used with bucket_by_length=True during adversary training.")
    common_args.add_argument('--disable_hf_caching', type=lambda x: bool(strtobool(x)), 
        help="If True, will not use the cache to reload the dataset.")
    common_args.add_argument('--n_shards', type=int,
        help="If above 0, will shard the dataset into that many shards. Used to get a small dataset for quick testing.")
    common_args.add_argument("--cache_dir", type=str, default="~/.cache/", help="Path to the cache directory.")
    
    ## Optimisation and training
    common_args.add_argument('--seed',          type=int,  help='Random seed')
    common_args.add_argument('--batch_size',      type=int, help='Batch size for training')
    common_args.add_argument('--batch_size_eval', type=int, help='Batch size for evaluation')
    common_args.add_argument('--learning_rate', type=float, help='Learning rate.')
    common_args.add_argument('--weight_decay',  type=float, help='Weight decay for AdamW')
    common_args.add_argument('--num_workers', type=int, help='Number of parallel worker threads for data loaders')
    common_args.add_argument('--early_stopping', type=lambda x: bool(strtobool(x)), 
                            help='If to do early stopping or not.')
    common_args.add_argument('--patience', type=int, help='Patience for early stopping.')
    common_args.add_argument("--optimizer_type", type=str, choices=['AdaFactor', 'AdamW'], help="Which optimiser to use.")
    ## Misc
    common_args.add_argument("--run_mode", type=str, required=True, choices=['dev', 'test', 'prod', 'sweep'],
        help="""Run mode. Dev is for development, test is for test runs on wandb, prod is for the actual experiments
        , and sweep is for hyperparameter sweeps.""")
    common_args.add_argument('--wandb_mode', type=str, choices=['online', 'disabled'],
         help='Set to "disabled" to suppress wandb logging.')
    common_args.add_argument('--log_to_stdout', type=lambda x: bool(strtobool(x)),
         help='Set to True to log to stdout as well as the log file.')
    return parser

def get_args_adversary():
    parser = ArgumentParser(description="Fine-tune a model on a text classification dataset.")    
    parser = Trainer.add_argparse_args(parser)  # Adds all lightning trainer args
    parser = add_common_args(parser)

    data_args = parser.add_argument_group('Data related arguments')
    data_args.add_argument('--long_example_behaviour', type=str,  
        choices=['remove', 'truncate'],
        help="Set to 'remove' to remove examples longer than `max_length_orig`, or 'truncate' to truncate them.")
    data_args.add_argument("--bucket_by_length", type=lambda x: bool(strtobool(x)),
        help="Set to True to load the data from longest-to-smallest (good for memory efficiency)")
    data_args.add_argument("--shuffle_buckets", type=lambda x: bool(strtobool(x)),
        help="Set to True to shuffle the buckets when bucket_by_length=True")
    data_args.add_argument("--use_preprocessed_data", type=lambda x: bool(strtobool(x)), 
        help="Set to True to use data that has already been preprocessed and saved to disk. Requires running the preprocessing script `adversary_dataset_preprocessing.py` first.")

    model_args = parser.add_argument_group('Model related arguments')
    model_args.add_argument('--pp_name',  type=str,                help="Name or path of paraphrase model to use.")
    model_args.add_argument('--ref_name', type=str,                help="Name or path of reference model to use, for KL div. Defaults to same as pp_name if not given.")
    model_args.add_argument('--vm_name',  type=str,                help="Name or path of victim model to use. If left blank, infers from dataset name and other models name.  ")
    model_args.add_argument('--sts_name', type=str,                help="Name or path of STS model to use. ")
    model_args.add_argument('--ld_name',  type=str,                help="Name or path of language detection model to use. ")
    model_args.add_argument("--use_peft", type=lambda x: bool(strtobool(x)), help="use a peft method (currently LoRa)")
    model_args.add_argument('--download_models', type=lambda x: bool(strtobool(x)), 
        help="If True, downloads the model from the HuggingFace model hub. Else, uses the cache.")
    model_args.add_argument('--freeze_vm_model', type=lambda x: bool(strtobool(x)), 
        help="If True, freezes the victim model so it doesn't update during training. Else, will also finetune parameters of the vm model.")
    misc_args = parser.add_argument_group('Logging and other misc arguments')
    misc_args.add_argument('--run_untrained', type=lambda x: bool(strtobool(x)),
         help='Set to True to evaluate val + test sets before we train to measure baseline untrained performance.')
    misc_args.add_argument('--delete_final_model', type=lambda x: bool(strtobool(x)),
         help='Set to True to delete the model after training (useful when running a sweep)')
    misc_args.add_argument("--hparam_profile", type=str, help="Preset hparam choices")

    parser = MultilingualWhiteboxAdversary.add_model_specific_args(parser) 
    args = parser.parse_args()

    # Set parameters as required 
    args = parse_vm_model_name(args)
    if args.ref_name is None: args.ref_name = args.pp_name  # if no ref model specified, use the pp model
    args = setup_defaults_adversary(args)
    if args.run_mode in ['dev', 'test']: delete_lockfiles()
    if   args.run_mode == 'dev':   args = setup_dev_mode_adversary(args)
    elif args.run_mode == 'test':  args = setup_test_mode_adversary(args)
    elif args.run_mode == 'prod':  args = setup_prod_mode_adversary(args)
    elif args.run_mode == 'sweep': args = setup_sweep_mode_adversary(args)
    else: raise Exception("shouldn't get here")
    args = setup_hparam_profiles_adversary(args)

    args = set_eval_gen_settings_from_eval_condition(args, args.eval_condition)
    if args.max_length_orig > 32 and args.use_preprocessed_data : 
        raise Exception("The preprocessing data script is currently set up to have a max length of 32. To have longer max length, disable this assert and reprocess the data.")
    return args

def get_args_victim_finetune(): 
    parser = ArgumentParser(description="Fine-tune a model on a text classification dataset.")    
    parser = Trainer.add_argparse_args(parser)  # Adds all lightning trainer args
    parser = add_common_args(parser)
    model_args = parser.add_argument_group('Model related arguments')
    model_args.add_argument('--model_name_or_path', type=str, help="Path to pretrained model or model identifier from huggingface.co/models")
    model_args.add_argument('--pp_name', type=str, help="Path to pretrained paraphrase or model identifier from huggingface.co/models")
    parser = VictimModel.add_model_specific_args(parser) 
    args = parser.parse_args()
    if args.model_name_or_path and args.pp_name:
        raise ValueError("Both --model_name_or_path and --pp_name were specified. Please specify only one.")
    args = setup_defaults_victim_finetuning(args)
    if   args.run_mode == 'dev':  args = setup_dev_mode_victim_finetuning(args)
    elif args.run_mode == 'test': args = setup_test_mode_victim_finetuning(args)
    elif args.run_mode == 'prod': args = setup_prod_mode_victim_finetuning(args)
    else: raise Exception("shouldn't get here")
    return args

def get_args_run_baselines(): 
    parser = ArgumentParser(description="Run baselines on datasets. ")   
    parser = Trainer.add_argparse_args(parser)  # Adds all lightning trainer args
    parser = add_common_args(parser)
    run_args = parser.add_argument_group('Run related arguments')
    run_args.add_argument('--vm_name',      type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
    run_args.add_argument('--attack_name',  type=str, required=True, help="What baseline attack to run.")
    run_args.add_argument('--query_budget',  type=int,  help="Max number of queries")

    args = parser.parse_args()
    args.freeze_vm_model = False 
    args.long_example_behaviour = "remove"
    # you have to update this to point to your paraphrase model
    args.pp_name = "/home/tproth/Data/model_checkpoints/multilingual_whitebox_finetuning/final/mt5_base_paraphrase.ckpt"
    args.bucket_by_length = False   # need the parameter set here for now to patch over some lazy code
    args.devices = [choose_gpu()] 
    args.num_workers= min(8, cpu_count()-1)
    args.cache_dir = "~/.cache/huggingface"  # update to point to your huggingface_cache_dir
    args.default_root_dir = ""
    args.num_sanity_val_steps = 0
    args.fast_dev_run = False
    args.batch_size = 5
    args.batch_size_eval=5

    args = parse_vm_model_name(args)
    if  args.run_mode == 'dev':  
        args.n_examples = 5
        args.max_length_orig = 10
        args.query_budget= 100
        args.wandb_mode = 'disabled'
    elif args.run_mode == 'test':
        args.n_examples = -1
        args.max_length_orig = 32
    return args

def parse_vm_model_name(args): 
    if "|" in args.vm_name: 
        # you can update this to point to a folder with models if you like. or just not use it
        args.vm_name = f"/home/tproth/Data/model_checkpoints/multilingual_whitebox_finetuning/final/{args.dataset_name}_{args.lang}.ckpt"
    return args

def set_eval_gen_settings_from_eval_condition(args, eval_condition): 
    assert eval_condition in ['standard', 'dev', 'ablation']
    gen_settings = {
        "val": {  # used in validation
            "num_return_sequences": 16,
            "num_beams": 16,
            "num_beam_groups": 8,
        }, 
        "bs_1": {
            "num_return_sequences": 1,
            "num_beams":16
        }, 
        "dbs_2": {
            "num_return_sequences": 2,
            "num_beams": 16,
            "num_beam_groups": 2,
        }, 
        "dbs_4": {
            "num_return_sequences": 4,
            "num_beams": 16,
            "num_beam_groups": 2,
        },
        "dbs_8": {
            "num_return_sequences": 8,
            "num_beams": 16,
            "num_beam_groups": 4,
        }, 
        "dbs_16": {
            "num_return_sequences": 16,
            "num_beams": 16,
            "num_beam_groups": 8,
        },
        "dbs_32": {
            "num_return_sequences": 32,
            "num_beams": 32,
            "num_beam_groups": 16,
        },
        "dbs_48": {
            "num_return_sequences": 48,
            "num_beams": 48,
            "num_beam_groups": 24,
        },
        "dbs_64": {
            "num_return_sequences": 64,
            "num_beams": 64,
            "num_beam_groups": 32,
        },
    }

    for k in gen_settings.keys():
        gen_settings[k]["temperature"] = 1.
        gen_settings[k]["top_p"] = 0.98
        gen_settings[k]["do_sample"] = False
        gen_settings[k]["return_dict_in_generate"] = True
        gen_settings[k]["output_scores"] = True
        if k != 'bs_1':  gen_settings[k]["diversity_penalty"] = 1.
    if   eval_condition == "standard": args.gen_settings = {'val': gen_settings['val'], 'dbs_32': gen_settings['dbs_32']}
    elif eval_condition == "dev":      args.gen_settings = {'val': gen_settings['val'], 'dbs_4':  gen_settings['dbs_4'], 'dbs_8':  gen_settings['dbs_8']}
    elif eval_condition == "ablation": args.gen_settings = gen_settings
    return args

def setup_hparam_profiles_adversary(args): 
    hparam_profiles = {
        "amazon": {
            "gumbel_tau": 1.10, 
            "coef_diversity": 0.10, 
            "coef_kl": 2.00, 
            "coef_sts": 20.00, 
            "coef_ld": 20.00, 
            "coef_vm": 15.00
        } ,
        "default": {
            "gumbel_tau": 1.10, 
            "coef_diversity": 10.00, 
            "coef_kl": 4.00, 
            "coef_sts": 25.00, 
            "coef_ld": 20.00, 
            "coef_vm": 15.00
        },
        "no_ld": {
            "gumbel_tau": 1.10, 
            "coef_diversity": 0.10, 
            "coef_kl": 2.00, 
            "coef_sts": 20.00, 
            "coef_ld": 0.00, 
            "coef_vm": 15.00
        } , 
        "no_ld_no_kl": {
            "gumbel_tau": 1.10, 
            "coef_diversity": 0.10, 
            "coef_kl": 0.00, 
            "coef_sts": 20.00, 
            "coef_ld": 0.00, 
            "coef_vm": 15.00
        } , 
        "no_kl": {
            "gumbel_tau": 1.10, 
            "coef_diversity": 0.10, 
            "coef_kl": 0.00, 
            "coef_sts": 20.00, 
            "coef_ld": 20.00, 
            "coef_vm": 15.00
        }  ,
        "no_sts": {
            "gumbel_tau": 1.10, 
            "coef_diversity": 0.10, 
            "coef_kl": 2.00, 
            "coef_sts": 0.00, 
            "coef_ld": 20.00, 
            "coef_vm": 15.00
        } , 
        "tweet_sentiment_multilingual": {
            "gumbel_tau": 1.10, 
            "coef_diversity": 10.00, 
            "coef_kl": 4.00, 
            "coef_sts": 25.00, 
            "coef_ld": 20.00, 
            "coef_vm": 15.00
        }  
    }
    if args.hparam_profile is not None: 
        args.gumbel_tau     = hparam_profiles[args.hparam_profile]["gumbel_tau"]
        args.coef_diversity = hparam_profiles[args.hparam_profile]["coef_diversity"]
        args.coef_kl        = hparam_profiles[args.hparam_profile]["coef_kl"]
        args.coef_sts       = hparam_profiles[args.hparam_profile]["coef_sts"]
        args.coef_ld        = hparam_profiles[args.hparam_profile]["coef_ld"]
        args.coef_vm        = hparam_profiles[args.hparam_profile]["coef_vm"]
    return args

def setup_defaults_adversary(args): 
    args.wandb_mode = 'disabled'

    # args.pp_name = "google/mt5-small"
    args.sts_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    args.ld_name = "DunnBC22/distilbert-base-multilingual-cased-language_detection"

    args.download_models = True
    args.freeze_vm_model = True
    # Hardware
    args.accelerator = "gpu"    
    args.cache_dir = "~/.cache/huggingface"  # huggingface cachedir

    args.num_workers= min(16, cpu_count()-1)

    # Data
    args.use_preprocessed_data = True
    args.min_length_orig = 0
    # args.min_length_pp = 4
    args.long_example_behaviour = "remove"
    args.bucket_by_length = True
    args.shuffle_train = False
    args.disable_hf_caching = False
    args.delete_final_model = True 
    args.shuffle_buckets = True

    # Training
    args.optimizer_type = "AdaFactor"
    args.learning_rate = 1e-5
    args.weight_decay = None

    args.use_peft = True
    # good defaults 
    args.eval_sts_threshold = 0.75 # intent: if sts    is below this value, add sts penalty, else leave it
    args.eval_ld_threshold = 0.0  # intent: if ld     is above this value, add ld penalty, else leave it
    args.eval_kl_threshold = 2.0  # intent: if kl div is above this value, add kl penalty, else leave it
    args.num_gumbel_samples = 5 

    # Misc
    args.default_root_dir = "../model_checkpoints/multilingual_whitebox"
    args.num_sanity_val_steps = 0
    args.fast_dev_run = False
    args.log_to_stdout = False
    args.profiler = None

    return args

def setup_dev_mode_adversary(args): 
    args.wandb_mode = 'disabled'
    # Datasets and models
    args.n_shards = 3
    # Paraphrase and orig parameters
    args.max_length_orig = 24
    # args.max_new_tokens_pp = 24
    
    args.run_untrained = False
    # args.eval_condition = "dev"
    args.eval_condition = "standard"

    # Batches and epochs
    args.batch_size = 4
    args.batch_size_eval = 4
    args.num_gumbel_samples = 3
    
    args.overfit_batches = 2
    args.limit_val_batches = 2
    args.limit_test_batches = 1
    args.max_epochs = 2
    args.early_stopping = True
    args.patience = 2 # any value will do 

    args.log_every_n_steps = 1
    args.check_val_every_n_epoch = 1


    # Debug on CPU
    # args.accelerator = "cpu"
    # args.devices = 1 
    # args.num_workers = 0 # min(8, cpu_count()-1)

    # Debug on GPU 
    args.accelerator = "gpu"    
    args.devices=[choose_gpu()]
    

   
    return args

def setup_test_mode_adversary(args): 
    args.run_untrained = False
    args.n_shards = 10

    # Paraphrase and orig parameters
    args.max_length_orig = 28
    args.num_gumbel_samples = 3
    args.eval_condition = "standard" 

    args.devices = [choose_gpu()]

    # Batches and epochs
    args.batch_size = 12
    args.batch_size_eval = 4
    args.learning_rate = 5e-4

    args.max_epochs = 3
    args.early_stopping = True
    args.patience = 10

    args.limit_val_batches = 8
    args.limit_test_batches = 8
    args.overfit_batches = 30  # must be bigger than val_check_interval
    args.val_check_interval = 5

    args.log_every_n_steps = 1
    return args    

def setup_prod_mode_adversary(args): 
    args.run_untrained = False
    args.n_shards = -1
    args.devices = [choose_gpu()]  
    args.delete_final_model = True
    args.use_peft = False
    # Paraphrase and orig parameters
    args.max_length_orig = 32
    args.eval_condition = "standard" 
    args.val_check_interval = 24

    # Batches and epochs
    args.batch_size = 5
    args.batch_size_eval = 5

    args.learning_rate =  1e-4

    # args.learning_rate =  2e-4
    args.early_stopping = True
    args.max_steps = 1400
    args.patience = 12
    args.log_every_n_steps = 5
    return args    

def setup_sweep_mode_adversary(args): 
    # select random seed as int between 100 and 9999
    args.seed = randint(100, 9999)

    args.run_untrained = False
    args.n_shards = -1
    args.devices = [get_least_occupied_gpu()] #[choose_gpu()]  # get_least_occupied_gpu()
    args.delete_final_model = True

    # Paraphrase and orig parameters
    args.num_gumbel_samples = 3
    args.gumbel_tau = 1.10
    args.max_length_orig = 28

    args.limit_val_batches = 20
    args.limit_train_batches = 100
    args.limit_test_batches = 40
    args.val_check_interval = 12


    # Batches and epochs
    args.batch_size = 14
    args.batch_size_eval = 6

    args.eval_condition = "standard" 
    args.learning_rate = 1e-4
    # args.max_epochs = 30
    args.max_steps = 126
    args.early_stopping = True
    args.patience = 4
    
    args.log_every_n_steps = 5
    return args    

def setup_defaults_victim_finetuning(args): 
    args.wandb_mode = 'disabled'
    # Models
    args.download_models = False

    # Hardware
    args.accelerator = "gpu"
    args.devices = [choose_gpu()] 
    args.num_workers= min(8, cpu_count()-1)
    args.cache_dir = "~/.cache/huggingface"  # huggingface cachedir

    # Data
    args.min_length_orig = 0
    args.shuffle_train = True
    args.disable_hf_caching = False
    args.bucket_by_length = False   # need the parameter set here for now to patch over some lazy code

    # Training
    args.seed = 9994
    args.optimizer_type = "AdamW"
    args.learning_rate = 0.0001
    args.weight_decay = 0.01

    # Misc
    args.default_root_dir = "../model_checkpoints/multilingual_whitebox_finetuning"
    args.num_sanity_val_steps = 0
    args.fast_dev_run = False
    return args

def setup_dev_mode_victim_finetuning(args): 
    args.wandb_mode = 'disabled'

    # Datasets and models
    args.n_shards = 100 

    # Paraphrase and orig parameters
    args.max_length_orig = 16

    # Batches and epochs
    args.batch_size = 4
    args.batch_size_eval = 4
    args.overfit_batches = 1
    args.max_epochs = 2
    args.early_stopping = False
    args.patience = 1

    args.log_every_n_steps = 1
    args.check_val_every_n_epoch = 1

    # Misc
    args.log_to_stdout = True
    return args

def setup_test_mode_victim_finetuning(args): 
    # Datasets and models
    args.n_shards = 10

    # Paraphrase and orig parameters
    args.max_length_orig = 48

    # Batches and epochs
    args.batch_size = 12
    args.batch_size_eval = 12

    args.overfit_batches = 1
    args.limit_val_batches= 1
    args.limit_test_batches=1

    args.max_epochs = -1
    args.num_sanity_val_steps = 0
    args.early_stopping = False
    args.patience = 1

    args.max_time="00:00:04:00" # 4 minutes 
    args.log_every_n_steps = 1
    args.check_val_every_n_epoch = 100
    # Misc
    args.log_to_stdout = True

    return args

def setup_prod_mode_victim_finetuning(args): 
    # Datasets and models
    args.n_shards = -1

    # Paraphrase and orig parameters
    args.max_length_orig = 48

    # Batches and epochs
    args.batch_size = 64
    args.batch_size_eval = 128
    args.max_epochs = -1
    args.num_sanity_val_steps = 0
    args.early_stopping = True
    args.patience = 8
    args.log_every_n_steps = 5
    args.val_check_interval = 12

    # Misc
    args.log_to_stdout = False
    return args

