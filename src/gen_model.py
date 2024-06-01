import torch
from pytorch_lightning import LightningModule

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from src.utils import gen_model_type
from pprint import pprint


class GenModel(LightningModule): 
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters() 
        self.args = args
        self.learning_rate = self.args.learning_rate

        ###### MODELS AND TOKENIZERS ######
        self.config = AutoConfig.from_pretrained(args.pp_name)
        self.tokenizer = AutoTokenizer.from_pretrained(args.pp_name)
        self.model  = AutoModelForSeq2SeqLM.from_pretrained(args.pp_name,  config=self.config)
        from src.models import get_pp_tokenizer_and_model
        self.pp_tokenizer,   self.pp_model   = get_pp_tokenizer_and_model(model_name_or_path=args.pp_name, args=args)

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def forward(self, batch):
        """Prediction/inference only"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        # decoder input ids should have shape (batch_size, target_sequence_length=1)
        ## If using t5 victim model, uncomment this
        if gen_model_type(self.args.pp_name) in ['mt5', 't5']:
            labels[labels == self.tokenizer.pad_token_id] = -100  # replace padding token id's of the labels by -100 so it's ignored by the loss
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        else: 
            # probably shouldn't get here
            raise Exception("Model type not supported, use a T5 or mT5 gen model.")
        return outputs
      
    def print_batch(self, batch, outputs): 
        inputs = self.tokenizer.batch_decode(batch['input_ids'])[0:6]
        labels = batch['labels'].clone()
        labels[labels == -100] = self.tokenizer.pad_token_id # -100 doesnt work for batch_decode
        label_text = self.tokenizer.batch_decode(labels)[0:6]
        oputs = self.tokenizer.batch_decode(torch.argmax(outputs['logits'], axis=2))[0:6]
        print("inputs: ")
        pprint(inputs)
        print('labels')
        pprint(label_text)
        print("outputs: ")
        pprint(oputs)
        del labels
        del label_text
        del oputs

    def training_step(self, batch, batch_idx):
        """complete training loop"""
        if not self.model.training: self.model.train()
        outputs = self(batch)
        loss = outputs.loss
        self.log('batch_loss', loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Complete validation loop"""
        if self.model.training: self.model.eval()
        with torch.no_grad(): 
            outputs = self(batch)
            if batch_idx == 0: 
                self.print_batch(batch, outputs)
        return outputs.loss

    def test_step(self, batch, batch_idx):
        """complete testing loop"""
        if self.model.training: self.model.eval()
        with torch.no_grad(): 
            outputs = self(batch)
            if batch_idx == 0: 
                self.print_batch(batch, outputs)
        return outputs.loss
    
    def predict_step(self, batch, batch_idx): 
        with torch.no_grad(): 
            outputs = self(batch)
        return outputs.loss

    def _log_epoch_end_metrics(self, outputs, key): 
        """Common logic for end of epoch metrics"""
        loss    = torch.stack(outputs).mean()
        self.log(f"{key}_loss", loss, prog_bar=True, on_epoch=True)
        
    def validation_epoch_end(self, outputs):
        self._log_epoch_end_metrics(outputs, key='val')

    def test_epoch_end(self, outputs):
        self._log_epoch_end_metrics(outputs, key='test')
    
    def configure_optimizers(self):
        """optimizers and LR schedulers"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        return optimizer
