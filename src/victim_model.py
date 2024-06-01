import torch
from wandb.plot import confusion_matrix
from pytorch_lightning import LightningModule

from evaluate import load
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from pprint import pprint
from src.utils import round_t
from torch.nn import CrossEntropyLoss
from src.dataset_prep import DS_INFO


class VictimModel(LightningModule): 
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters() 
        self.args = args
        self.learning_rate = self.args.learning_rate
        self.dataset_num_labels = DS_INFO[args.dataset_name]['num_labels']
        ###### MODELS AND TOKENIZERS ######
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=self.dataset_num_labels)
        self.metric_precision  = load("precision")
        self.metric_accuracy   = load("accuracy")
        self.metric_recall     = load("recall")
        self.metric_f1         = load("f1")

    @staticmethod
    def add_model_specific_args(parent_parser):
        return parent_parser

    def forward(self, input_ids, attention_mask):
        """Prediction/inference only"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits

    def _forward_and_calc_metrics(self, batch, predict=False): 
        """Common logic for model predictions"""
        logits = self(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        if len(batch['label']) == 1: logits = logits.unsqueeze(0)  # problems with last batches of size 1 
        preds = torch.argmax(logits, axis=1)
        if not predict: 
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(logits, batch['label'])
            results = {"loss": loss, "preds": preds, "logits": logits, "labels": batch['label']}
        else: 
            results = {"preds": preds, "logits": logits}
        return results

    def training_step(self, batch, batch_idx):
        """complete training loop"""
        if self.model.eval(): self.model.train()
        results = self._forward_and_calc_metrics(batch)
        self.log('batch_loss', results['loss'], prog_bar=True, on_step=True)
        if self.current_epoch % 10 == 0 and batch_idx <= 2: 
            self.print_batch(batch, results)
        return results

    def print_batch(self, batch, results): 
        inputs = self.tokenizer.batch_decode(batch['input_ids'])[0:8]
        pprint('input text')
        pprint(inputs)
        pprint('labels: ')
        pprint(batch['label'][0:8])
        pprint('probs') 
        pprint(round_t(torch.softmax(results['logits'][0:8], dim=-1)))
        pprint ("batch_loss: ")
        pprint(results['loss'])
        del inputs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Complete validation loop"""
        self.model.eval()
        with torch.no_grad():
            results = self._forward_and_calc_metrics(batch)
        return results

    def test_step(self, batch, batch_idx):
        """complete testing loop"""
        self.model.eval()
        with torch.no_grad():
            results = self._forward_and_calc_metrics(batch)
        return results
    
    def predict_step(self, batch, batch_idx): 
        self.model.eval()
        with torch.no_grad():
            results = self._forward_and_calc_metrics(batch, predict=True)
        return results

    def _log_epoch_end_metrics(self, outputs, key): 
        """Common logic for end of epoch metrics"""
        preds   = torch.cat(  [x["preds"]  for x in outputs]).detach().cpu().numpy()
        labels  = torch.cat(  [x["labels"] for x in outputs]).detach().cpu().numpy()
        loss    = torch.stack([x["loss"]   for x in outputs]).mean()
        self.log(f"{key}_loss", loss, prog_bar=True, on_epoch=True)
        
        # Accuracy 
        d = self.metric_accuracy.compute(predictions=preds, references=labels)
        d[f'{key}_accuracy'] = d.pop('accuracy')
        self.log_dict(d, on_epoch = True, prog_bar=True)

        # Precision, recall
        for metric_fn in [self.metric_precision, self.metric_recall]: 
            d = metric_fn.compute(predictions=preds, references=labels , average='weighted', zero_division=0) # average needed for multiclass classification
            metric_name = list(d.keys())[0]
            d[f'{key}_{metric_name}'] = d.pop(metric_name)
            self.log_dict(d, on_epoch = True, prog_bar=True)

        # F1
        d = self.metric_f1.compute(predictions=preds, references=labels, average='weighted')
        d[f'{key}_f1'] = d.pop('f1')
        self.log_dict(d, on_epoch = True, prog_bar=True)

        # Confusion matrix
        if key in ['val', 'test']: 
            cm = confusion_matrix(preds=preds, y_true=labels, class_names=list(DS_INFO[self.args.dataset_name]['LABEL2ID'].keys()))
            self.logger.experiment.log({f'confusion_matrix_{key}': cm})

    def training_epoch_end(self, outputs):
        self._log_epoch_end_metrics(outputs, key='train')

    def validation_epoch_end(self, outputs):
        self._log_epoch_end_metrics(outputs, key='val')

    def test_epoch_end(self, outputs):
        self._log_epoch_end_metrics(outputs, key='test')
    
    def configure_optimizers(self):
        """optimizers and LR schedulers"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.args.weight_decay)
        return optimizer
