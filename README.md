
Code for the paper [A Generative Adversarial Attack for Multilingual Text Classifiers](https://arxiv.org/abs/2401.08255), which has the abstract

> Current adversarial attack algorithms, where an adversary changes a text to fool a victim model, have been repeatedly shown to be effective against text classifiers. These attacks, however, generally assume that the victim model is monolingual and cannot be used to target multilingual victim models, a significant limitation given the increased use of these models. For this reason, in this work we propose an approach to fine-tune a multilingual paraphrase model with an adversarial objective so that it becomes able to generate effective adversarial examples against multilingual classifiers. The training objective incorporates a set of pre-trained models to ensure text quality and language consistency of the generated text. In addition, all the models are suitably connected to the generator by vocabulary-mapping matrices, allowing for full end-to-end differentiability of the overall training pipeline. The experimental validation over two multilingual datasets and five languages has shown the effectiveness of the proposed approach compared to existing baselines, particularly in terms of query efficiency. We also provide a detailed analysis of the generated attacks and discuss limitations and opportunities for future research.


## Installation 

To run yourself, create a virtual environment (using whatever tool you prefer) and install the packages at `environment.yml` file, which contains a complete list of packages used in the project. You probably don't need all of these, so for a smaller environment, try starting from a basic python install with the following versions of some key packages:

* python=3.10.9
* pandas==1.5.2
* torch==2.1.0+cu118
* tokenizers==0.13.4
* wandb==0.15.0
* sentence-transformers==2.2.2
* datasets==2.15.0
* evaluate==0.4.0
* pytorch-lightning==1.9.0


Some of these scripts support weights and biases (wandb). If you don't wish to use it, pass `--wandb_mode disabled` when calling scripts. Else, pass `--wandb_mode online` and adjust the code to use your entity name and project name. 

## Preprocessing

Run `python adversary_dataset_preprocessing.py` to preprocess the data (can take a while). You can either set parameters yourself in the script, or pass them into the function as arguments when running from the command line. The important parameters: `dataset_name`, `lang`, `pp_name`. 

To finetune a classifier on a dataset to then attack run something like

```
python victim_finetune.py --model_name_or_path distilbert-base-multilingual-cased --dataset_name tweet_sentiment_multilingual --lang ar --run_mode test 
 ```
adjusting the parameters as you like. 

To finetine the mt5 model on the paraphrasing seq2seq task, run something like 

```
python gen_finetune.py --pp_name google/mt5-small --lang all --run_mode test 
```
specifying the base model as `pp_name`. See the code for all parameters this takes.


### Adversary

Afterwards you can run `training.py` to run the adversary on the processed dataset and against the trained victim mdoel. For example

```
python training.py --dataset_name tweet_sentiment_multilingual --lang all --pp_name mt5_small_paraphrase.ckpt --ref_name mt5_small_paraphrase.ckpt --vm_name victim_model.ckpt --hparam_profile default --seed 1000 --run_mode dev --download_models False 
```
There are many parameters and see the code `src/parsing_fns.py` for documentation of them. 

Note: the `amazon_reviews_multi` dataset has now been removed from HuggingFace as "Amazon has decided to stop distributing the multilingual reviews dataset", which is great news for every non-Amazon researcher. So the code now only works with the  `tweet_sentiment_multilingual` dataset. 

### Baselines
Run the baselines with

```
python run_baselines.py --vm_name model.ckpt --dataset_name tweet_sentiment_multilingual --run_mode test --attack_name bae_multilingual --accelerator gpu --lang all 
```
adjusting as needed. 