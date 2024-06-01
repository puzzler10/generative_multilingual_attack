
import pickle
from pprint import pprint
import os
from pytorch_lightning import Trainer
from src.parsing_fns import get_args_adversary
from src.training_fns import load_preprocessed_dataset
from src.adversary import MultilingualWhiteboxAdversary
from src.utils import *
from pytorch_lightning.utilities import rank_zero_only

def main(args): 
    from src.training_fns import setup_loggers,setup_environment,load_dataset,get_callbacks 
    setup_environment(args)
    project = "your_project_here"  # update with your wandb project name here 
    wandb_logger, path_run = setup_loggers(args, project=project)
    adversary = MultilingualWhiteboxAdversary(args) 
    adversary.path_run = path_run
    if args.use_preprocessed_data:
        dataset = load_preprocessed_dataset(args)
    else: 
        dataset = load_dataset(args, project, pp_tokenizer=adversary.pp_tokenizer, vm_tokenizer=adversary.vm_tokenizer, ld_tokenizer=adversary.ld_tokenizer,
                          vm_model=adversary.vm_model, sts_model=adversary.sts_model, ld_model=adversary.ld_model)
    try: 
        del dataset.vm_model
        del dataset.sts_model
    except AttributeError: 
        pass

    # cache arguments
    with open(f'{args.cache_dir}/args_cached.pickle', 'wb')    as handle: pickle.dump(args,    handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Adversary sts model seems to keep random parameters updateable 
    for name, param in adversary.sts_model.named_parameters():
        if param.requires_grad: param.requires_grad = False
    
    # Save our processed datasets to csv to join with the training data later 
    for k, ds in dataset.dsd.items(): 
        ds.remove_columns(['orig_ids_pp_tknzr', 'attention_mask_pp_tknzr', 'orig_sts_embeddings']).to_csv(f"{path_run}/orig_{k}.csv",index=False)

    callbacks = get_callbacks(args, metric="adv_score_validation_mean", mode="max")

    trainer = Trainer.from_argparse_args(args,
        logger=wandb_logger, 
        default_root_dir=path_run, 
        callbacks=callbacks
    )
    train_dataloaders,val_dataloaders,test_dataloaders=dataset.dld['train'],dataset.dld['validation'],dataset.dld['test']
    assert len(train_dataloaders) > 0 and len(val_dataloaders) > 0 and len(test_dataloaders) > 0, "A dataloader is empty"
    
    if args.run_untrained:
        adversary.untrained_run = True
        test_results = trainer.test(model=adversary,  dataloaders=test_dataloaders,  verbose=True)
        if rank_zero_only.rank == 0:
            if len(test_results) > 1:
                wandb_logger.experiment.config.update({f"untrained_{k}":v for k,v in test_results[0].items()})
            else: 
                print("Test results didn't have values")

    adversary.untrained_run = False
    trainer.fit( model=adversary, train_dataloaders=train_dataloaders, val_dataloaders=val_dataloaders) 
    adversary.best_global_step  = int(float(trainer.checkpoint_callback.best_model_path.split('step=')[1].split('-')[0]))
    wandb_logger.experiment.summary[f"best_global_step"] = adversary.best_global_step
    test_dl = val_dataloaders if args.run_mode == 'sweep' else test_dataloaders
    trainer.test(model=adversary,  dataloaders=test_dl, ckpt_path="best", verbose=True)
    if args.delete_final_model: os.remove(trainer.checkpoint_callback.best_model_path)
 
if __name__ == "__main__":
    args = get_args_adversary()
    pprint(vars(args))
    main(args)