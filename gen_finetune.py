
from pprint import pprint
import logging
import pytorch_lightning as pl
from transformers.utils.versions import require_version
import src.parsing_fns 
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")
logger = logging.getLogger(__name__)

def main(args): 
    from src.gen_model import GenModel
    from src.training_fns import setup_loggers,setup_environment,load_dataset,get_callbacks

    setup_environment(args)
    project = "multilingual_gen_finetune"
    wandb_logger, path_run = setup_loggers(args, project=project) 
    model = GenModel(args)
    model.path_run = path_run
    dataset = load_dataset(args, project, tokenizer=model.tokenizer)
    
    callbacks = get_callbacks(args, metric="val_loss", mode="min")
    trainer = pl.Trainer.from_argparse_args(args,
        logger=wandb_logger, 
        default_root_dir=path_run, 
        callbacks=callbacks
    )
    train_dataloaders,val_dataloaders,test_dataloaders=dataset.dld['train'],dataset.dld['validation'],dataset.dld['test']    
    trainer.fit(model=model, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders) 
    trainer.test(model, ckpt_path="best", verbose=True, dataloaders=test_dataloaders)    

    

if __name__ == "__main__":
    args = src.parsing_fns.get_args_victim_finetune()
    pprint(vars(args))
    main(args)