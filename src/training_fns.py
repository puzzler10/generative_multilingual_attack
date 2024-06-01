import pickle
import logging
import os
import warnings
from pytorch_lightning import seed_everything
from datasets import config
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from src.dataset_prep import AdversaryDataset, VictimFineTuningDataset, GenFineTuningDataset,BaselineDataset
from src.utils import file_is_unchanged
import fcntl
from socket import gethostname
from torch.cuda import device_count


logger = logging.getLogger(__name__) #

DATASET_ARGS =  ['accelerator', 'auto_scale_batch_size', 'auto_select_gpus', 'batch_size', 'batch_size_eval', 'bucket_by_length', 'dataset_name', 
      'devices', 'disable_hf_caching', 'gpus', 'ld_name', 'lang','long_example_behaviour', 'max_length_orig', 'min_length_orig', 
       'multiple_trainloader_mode', 'model_name_or_path', 'n_shards', 'num_nodes', 'num_proceses', 'num_workers', 'overfit_batches', 'pp_name', 'precision', 
       'reload_dataloaders_every_n_epochs', 'run_mode', 'seed', 'shuffle_buckets', 'shuffle_train', 'strategy', 'sts_name', 'vm_name']


def setup_environment(args): 
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    if   args.run_mode in ['dev', 'test']:            os.environ['DEBUG_MODE'] = 'true'
    elif args.run_mode in ['prod', 'sweep']:          os.environ['DEBUG_MODE'] = 'false'
    warnings.filterwarnings("ignore", message="Passing `max_length` to BeamSearchScorer is deprecated")  # we ignore the warning because it works anyway for diverse beam search 
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    seed_everything(seed=args.seed)
    config.IN_MEMORY_MAX_SIZE = 5368709120 # keep datasets under 5GB in memory
    setup_cache_dir(args)

def choose_gpu():
    hostname = gethostname()
    for device_id in range(device_count()):
        lockfile_path = f"/tmp/{hostname}_gpu_{device_id}.lock"
        try:
            lockfile = os.open(lockfile_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            # Try to lock the file. This will fail if another process has the file locked.
            fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return device_id
        except OSError as e:
            # File is locked (possibly by another process)
            print(f"Device {device_id} is in use.")
            continue
    print("No available GPUs, defaulting to device 0")
    return 0 

def delete_lockfiles(): 
    """Manually remove lockfiles for debug mode."""
    import glob, socket
    hostname = socket.gethostname()
    lockfiles = glob.glob(f'/tmp/{hostname}_gpu_*')
    for file in lockfiles:
        os.remove(file)

def occupy_gpu_mem(args):
    cuda_device =  str(args.devices[0])
    os.environ["CUDA_VISIBLE_DEVICES"] =  cuda_device
    def check_mem(cuda_device):
        devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
        total, used = devices_info[int(cuda_device)].split(',')
        return total,used
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    from torch.cuda import FloatTensor
    x = FloatTensor(256,1024,block_mem)
    del x


def get_device_string(args): 
    if   args.accelerator == 'cpu': return 'cpu'
    elif args.accelerator == "gpu": return f'cuda:{args.devices[0]}'
    else: raise Exception('unspecified device config')

def setup_cache_dir(args):
    """Use scratch directory as the datasets cache."""
    dirs = [f'{args.cache_dir}/', f'{args.cache_dir}/hf_datasets_cache/']
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = dirs[1]


def setup_loggers(args, project): 
    # update this with your entity if using wandb
    wandb_logger = WandbLogger(project=project, entity="your_entity_here", save_dir=args.default_root_dir, mode=args.wandb_mode)
    path_run = f"{args.default_root_dir}/{wandb_logger.experiment.name}"
    if not os.path.exists(path_run): os.makedirs(path_run, exist_ok=True)
    log_filename = path_run + "/log.txt"
    handlers = [logging.FileHandler(log_filename)]
    if args.log_to_stdout: handlers += [logging.StreamHandler()]
    logging.basicConfig(handlers=handlers, level=logging.INFO)
    print(f"Files and models are logged in folder {path_run}")
    print(f"Log file: {log_filename}")
    return wandb_logger, path_run


def get_dset_fname_from_args(args): 
    """Used for saving and loading preprocessed dataset. returns string with filename for the preprocessed data."""
    # strip path and extension from pp_name if it exists 
    pp_name_processed = args.pp_name
    if '/' in pp_name_processed:
        pp_name_processed = pp_name_processed.split('/')[-1]
    if '.' in pp_name_processed:
        pp_name_processed = pp_name_processed.split('.')[0]
    def make_filename_safe(s):   return s.replace("/", "_").replace("-", "_").lower()
    sts_name_safe = make_filename_safe(args.sts_name)
    ld_name_safe  = make_filename_safe(args.ld_name)
    pp_name_safe  = make_filename_safe(pp_name_processed)
    fname = f"data/adversary_datasets/DS--{args.dataset_name}--LANG--{args.lang}--PP--{pp_name_safe}--STS--{sts_name_safe}--LD--{ld_name_safe}.pkl"
    return fname


def args_havent_changed(args, ARGSET):
    try:
        with open(f'{args.cache_dir}/args_cached.pickle', 'rb') as handle: 
            old_args = pickle.load(handle)
            # check if any dataset args have changed
            args_are_equal = all(getattr(args, ds_arg, None) == getattr(old_args, ds_arg, None) for ds_arg in ARGSET)
            return args_are_equal
    except FileNotFoundError:
        print(f"No previous args found in cache so cache not used.")
        return False
    
def load_preprocessed_dataset(args, fname=None): 
    if fname is None: fname = get_dset_fname_from_args(args)
    logger.info(f"Loading preprocessed dataset from {fname}")
    with open(fname, 'rb') as f:
        dataset = pickle.load(f)
    # If max_orig_length is less than 32, filter out those bigger than that 
    if args.max_length_orig < 32:
        if args.long_example_behaviour == 'remove': 
            dataset.dsd = dataset.dsd.filter(lambda x: x['n_tokens'] <= args.max_length_orig)
    dataset.dld = dataset.prep_dataloaders(dsd=dataset.dsd,args=args)
    return dataset


def load_dataset(args, project, **kwargs): 
    if os.environ['DEBUG_MODE'] == 'true' and file_is_unchanged("src/dataset_prep.py", args) and args_havent_changed(args, DATASET_ARGS):
        print("Loading cached dataset.")
        with open(f'{args.cache_dir}/dataset_cached.pickle', 'rb') as handle: dataset = pickle.load(handle)
    else: 
        if      "multilingual_victim_finetune" in project:                   DatasetClass = VictimFineTuningDataset
        elif    "multilingual_gen_finetune"    in project:                   DatasetClass = GenFineTuningDataset
        elif    "multilingual_whitebox"        in project:                   DatasetClass = AdversaryDataset
        elif    "multilingual_baselines"       in project:                   DatasetClass = BaselineDataset
        else: raise ValueError("Project name not recognised.")
        dataset = DatasetClass(args, **kwargs)
        if os.environ['DEBUG_MODE'] == 'true': 
            with open(f'{args.cache_dir}/dataset_cached.pickle', 'wb') as handle: pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return dataset


def get_callbacks(args, metric, mode): 
    callbacks = [
        ModelCheckpoint(monitor=metric, save_top_k=1, mode=mode, 
                        verbose=True, 
                        filename='{step}-{adv_score_validation_mean:.3f}', 
                        save_weights_only=True)
    ]
    if args.early_stopping: callbacks.append(EarlyStopping(monitor=metric, mode=mode, patience=args.patience))
    return callbacks

def get_profiler(args): 
    from pytorch_lightning.profilers import AdvancedProfiler, SimpleProfiler, PyTorchProfiler
    if   args.profiler == "simple": 
        profiler = SimpleProfiler(dirpath=".", filename="perf_logs_simple")
    elif args.profiler == "advanced": 
        profiler = AdvancedProfiler(dirpath=".", filename="perf_logs_adv")
    elif args.profiler == "pytorch": 
        profiler = PyTorchProfiler(dirpath=".", filename="perf_logs_pt", export_to_chrome=False)
    elif args.profiler is None: 
        profiler = None
    return profiler

def log_best_global_step_to_wandb(trainer, wandb_logger): 
    best_global_step = trainer.checkpoint_callback.best_model_path.split('epoch=')[1].split('-')[0]
    trainer.best_global_step = best_global_step
    wandb_logger.experiment.summary[f"best_global_step"] = best_global_step
    