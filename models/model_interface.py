import os
import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd

#---->
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from utils.utils import NLLSurvLoss, CrossEntropySurvLoss, cox_log_rank, _predictions_to_pycox, CoxSurvLoss
from sksurv.metrics import concordance_index_censored
# from pycox.evaluation import EvalSurv
from sklearn.utils import resample


#---->
import torch
import torch.nn as nn
import torch.nn.functional as F

#---->
import pytorch_lightning as pl



import os
import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd

#---->

from sksurv.metrics import concordance_index_censored
# from pycox.evaluation import EvalSurv
from sklearn.utils import resample


#---->
import torch
import torch.nn as nn
import torch.nn.functional as F

#---->
import pytorch_lightning as pl

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc as calc_auc, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
from colorama import Fore, Style, init


def loss_reg_l1(coef=0.00001):
    print('[setup] L1 loss with coef={}'.format(coef))
    coef = .0 if coef is None else coef
    def func(model_params):
        if coef <= 1e-8:
            return 0.0
        else:
            return coef * sum([torch.abs(W).sum() for W in model_params])
    return func

# Initialize colorama for colored terminal output
init(autoreset=True)

class ModelInterfaceProg(pl.LightningModule):

    # ---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterfaceProg, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        self.loss = NLLSurvLoss(loss.alpha_surv)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']

        self.training_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []

        # Print initialization information
        print(Fore.CYAN + "═" * 80)
        print(f"{Fore.GREEN}Model initialized: {Fore.YELLOW}ModelInterfaceProg{Style.RESET_ALL}")
        print(f"  Loss: {Fore.BLUE}NLLSurvLoss(alpha={loss.alpha_surv}){Style.RESET_ALL}")
        print(f"  Model: {Fore.BLUE}{model.name}{Style.RESET_ALL}")
        print(f"  Log path: {Fore.BLUE}{self.log_path}{Style.RESET_ALL}")
        print(Fore.CYAN + "═" * 80)

    def training_step(self, batch, batch_idx):
        if len(batch) == 4:
            data_WSI, label, event_time, c = batch
            results_dict = self.model(data=data_WSI)
        else:
            sample, survival = batch
            label = survival[0][0].unsqueeze(0)
            event_time = survival[0][1].unsqueeze(0)
            c = survival[0][2].unsqueeze(0)
            results_dict = self.model(data=sample)

        hazards = results_dict['hazards']
        S = results_dict['S']
        Y_hat = results_dict['Y_hat']

        # ---->calculate loss
        loss = self.loss(hazards=hazards, S=S, Y=label.long(), c=c)
        self.training_step_outputs.append({
            'loss': loss.item(),
            'Y_hat': Y_hat,
            'label': label,
            'S': S
        })
        return loss

    def on_train_epoch_end(self):
        current_epoch = self.current_epoch + 1
        training_step_outputs = self.training_step_outputs

        # Calculate average loss
        all_losses = [x['loss'] for x in training_step_outputs]
        avg_loss = np.mean(all_losses)

        # Log metrics
        self.log('train_loss', avg_loss, on_epoch=True, logger=True)

        # Beautified output
        print(Fore.CYAN + "═" * 80)
        print(Fore.GREEN + f"【TRAINING RESULTS】- Epoch: {Fore.YELLOW}{current_epoch}{Style.RESET_ALL}")
        print(f"  Average Loss: {Fore.YELLOW}{avg_loss:.4f}{Style.RESET_ALL}")
        print(f"  Samples: {Fore.BLUE}{len(training_step_outputs)}{Style.RESET_ALL}")
        print(Fore.CYAN + "═" * 80)

        self.training_step_outputs = []

    def validation_step(self, batch, batch_idx):
        if len(batch) == 4:
            data_WSI, label, event_time, c = batch
            results_dict = self.model(data=data_WSI)
        else:
            sample, survival = batch
            label = survival[0][0].unsqueeze(0)
            event_time = survival[0][1].unsqueeze(0)
            c = survival[0][2].unsqueeze(0)
            results_dict = self.model(data=sample)

        hazards = results_dict['hazards']
        S = results_dict['S']
        Y_hat = results_dict['Y_hat']

        loss = self.loss(hazards=hazards, S=S, Y=label.long(), c=c)
        risk = -torch.sum(S, dim=1).cpu().item()
        self.val_step_outputs.append({
            'loss': loss.item(),
            'risk': risk,
            'censorship': c.item(),
            'event_time': event_time.item()
        })
        return {'val_loss': loss}

    def on_validation_epoch_end(self):
        current_epoch = self.current_epoch + 1
        val_step_outputs = self.val_step_outputs
        all_val_loss = np.stack([x['loss'] for x in val_step_outputs])
        all_risk_scores = np.stack([x['risk'] for x in val_step_outputs])
        all_censorships = np.stack([x['censorship'] for x in val_step_outputs])
        all_event_times = np.stack([x['event_time'] for x in val_step_outputs])

        # Calculate metrics
        mean_loss = np.mean(all_val_loss)
        c_index = concordance_index_censored(
            (1 - all_censorships).astype(bool),
            all_event_times,
            all_risk_scores,
            tied_tol=1e-08
        )[0]
        pvalue_pred = cox_log_rank(all_risk_scores, (1 - all_censorships), all_event_times)

        # Log metrics
        self.log('val_loss', mean_loss, prog_bar=True, on_epoch=True, logger=True)
        self.log('c_index', c_index, prog_bar=True, on_epoch=True, logger=True)
        self.log('p_value', pvalue_pred, prog_bar=True, on_epoch=True, logger=True)

        # Calculate events and censored counts
        events_count = np.sum((1 - all_censorships).astype(bool))
        censored_count = np.sum(all_censorships.astype(bool))

        # Beautified output
        print(Fore.CYAN + "═" * 80)
        print(Fore.MAGENTA + f"【VALIDATION RESULTS】- Epoch: {Fore.YELLOW}{current_epoch}{Style.RESET_ALL}")
        print(f"  Val Loss:    {Fore.YELLOW}{mean_loss:.4f}{Style.RESET_ALL}")
        print(f"  C-Index:     {Fore.YELLOW}{c_index:.4f}{Style.RESET_ALL}")
        print(f"  P-Value:     {Fore.YELLOW}{pvalue_pred:.6f}{Style.RESET_ALL}")
        print(
            f"  Data:        {Fore.BLUE}{len(all_risk_scores)} samples ({events_count} events, {censored_count} censored){Style.RESET_ALL}")
        print(Fore.CYAN + "═" * 80)

        self.val_step_outputs = []

    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]

    def test_step(self, batch, batch_idx):
        if len(batch) == 4:
            data_WSI, label, event_time, c = batch
            results_dict = self.model(data=data_WSI)
        else:
            sample, survival = batch
            label = survival[0][0].unsqueeze(0)
            event_time = survival[0][1].unsqueeze(0)
            c = survival[0][2].unsqueeze(0)
            results_dict = self.model(data=sample)

        hazards = results_dict['hazards']
        S = results_dict['S']
        Y_hat = results_dict['Y_hat']

        risk = -torch.sum(S, dim=1).cpu().item()

        self.test_step_outputs.append({
            'risk': risk,
            'censorship': c.item(),
            'event_time': event_time.item(),
            'S': S.cpu().detach() if 'S' in results_dict else None
        })

    def on_test_epoch_end(self):
        test_step_outputs = self.test_step_outputs
        all_risk_scores = np.stack([x['risk'] for x in test_step_outputs])
        all_censorships = np.stack([x['censorship'] for x in test_step_outputs])
        all_event_times = np.stack([x['event_time'] for x in test_step_outputs])

        # Check if S values exist in outputs
        has_properties = all(x['S'] is not None for x in test_step_outputs)
        if has_properties:
            all_properties = torch.cat([x['S'] for x in test_step_outputs]).cpu().detach()
        else:
            all_properties = None

        # Calculate metrics
        c_index = concordance_index_censored(
            (1 - all_censorships).astype(bool),
            all_event_times,
            all_risk_scores,
            tied_tol=1e-08
        )[0]
        pvalue_pred = cox_log_rank(all_risk_scores, (1 - all_censorships), all_event_times)

        # Calculate events and censored counts
        events_count = np.sum((1 - all_censorships).astype(bool))
        censored_count = np.sum(all_censorships.astype(bool))

        # Beautified initial output
        print(Fore.CYAN + "═" * 80)
        print(Fore.RED + "【TEST RESULTS】" + Style.RESET_ALL)
        print(f"  C-Index:     {Fore.YELLOW}{c_index:.4f}{Style.RESET_ALL}")
        print(f"  P-Value:     {Fore.YELLOW}{pvalue_pred:.6f}{Style.RESET_ALL}")
        print(
            f"  Data:        {Fore.BLUE}{len(all_risk_scores)} samples ({events_count} events, {censored_count} censored){Style.RESET_ALL}")
        print(Fore.CYAN + "═" * 80)

        # ---->bootstrap analysis
        print(Fore.BLUE + f"Running bootstrap analysis ({Fore.YELLOW}1000 iterations{Fore.BLUE})..." + Style.RESET_ALL)
        n = 1000
        skipped = 0
        boot_c_index = []

        from tqdm import tqdm
        for i in tqdm(range(n), desc="Bootstrap Progress", ncols=100, colour="green"):
            boot_ids = resample(np.arange(len(all_risk_scores)), replace=True)
            risk_scores = all_risk_scores[boot_ids]
            censorships = all_censorships[boot_ids]
            event_times = all_event_times[boot_ids]

            if has_properties:
                properties = all_properties[boot_ids]

            # When running samples with small number of patients, sometimes there are no admissible pairs
            try:
                c_index_buff = concordance_index_censored(
                    (1 - censorships).astype(bool),
                    event_times,
                    risk_scores,
                    tied_tol=1e-08
                )[0]
                boot_c_index.append(c_index_buff)
            except ZeroDivisionError as error:
                err = error
                skipped += 1
                continue

        if skipped > 0:
            warnings.warn(f'Skipped {skipped} bootstraps ({err}).')

        # ---->Calculate confidence intervals
        c_index_differences = sorted([x - c_index for x in boot_c_index])
        c_index_percent = np.percentile(c_index_differences, [2.5, 97.5])
        c_index_low, c_index_high = tuple(round(c_index + x, 4) for x in [c_index_percent[0], c_index_percent[1]])

        # Bootstrap results output
        print(Fore.CYAN + "═" * 80)
        print(Fore.BLUE + "【BOOTSTRAP RESULTS】" + Style.RESET_ALL)
        print(f"  C-Index CI:  {Fore.YELLOW}{c_index_low} - {c_index_high}{Style.RESET_ALL}  (95% confidence interval)")
        if skipped > 0:
            print(f"  Warning:     {Fore.RED}Skipped {skipped} bootstraps due to insufficient data{Style.RESET_ALL}")
        print(Fore.CYAN + "═" * 80)

        # ---->Save results
        print(f"{Fore.GREEN}Saving results to: {Fore.BLUE}{self.log_path}{Style.RESET_ALL}")

        # Create log directory if it doesn't exist
        os.makedirs(self.log_path, exist_ok=True)

        # Save metrics as CSV
        metrics_dict = {
            'c_index': c_index,
            'c_index_high': c_index_high,
            'c_index_low': c_index_low,
            'p_value': pvalue_pred,
            'samples': len(all_risk_scores),
            'events': events_count,
            'censored': censored_count
        }
        result = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'Value'])
        result.to_csv(os.path.join(self.log_path, 'result.csv'), index=False)

        # Also save as a more readable format
        pd.DataFrame([metrics_dict]).to_csv(os.path.join(self.log_path, 'metrics.csv'), index=False)

        # ---->Save the risk scores, censorships, and event times
        np.savez(os.path.join(self.log_path, 'all_risk_scores.npz'), all_risk_scores)
        np.savez(os.path.join(self.log_path, 'all_censorships.npz'), all_censorships)
        np.savez(os.path.join(self.log_path, 'all_event_times.npz'), all_event_times)

        # Also save as CSV for easier access
        survival_df = pd.DataFrame({
            'risk_score': all_risk_scores,
            'censorship': all_censorships,
            'event_time': all_event_times
        })
        survival_df.to_csv(os.path.join(self.log_path, 'survival_data.csv'), index=False)

        print(f"{Fore.GREEN}Results saved successfully.{Style.RESET_ALL}")
        print(Fore.CYAN + "═" * 80)

        self.test_step_outputs = []

    def load_model(self):
        name = self.hparams.model.name
        # Convert snake_case to CamelCase for class name
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(f'models.{name}'), camel_name)
            print(f"{Fore.GREEN}Model loaded successfully: {Fore.CYAN}{camel_name}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error loading model: {str(e)}{Style.RESET_ALL}")
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)


class ModelInterfaceCls(pl.LightningModule):

    # ---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterfaceCls, self).__init__()
        self.save_hyperparameters()
        self.load_model()
        print(f"Loss function: {loss}")
        self.loss = create_loss(loss)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']

        self.training_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []

    def training_step(self, batch, batch_idx):
        # ---->inference
        data, label = batch
        results_dict = self.model(data=data, is_train=True, label=label)

        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        loss = 0
        if 'attn_score' in results_dict:
            instance_attn_score = results_dict['attn_score']
            instance_attn_score_normed = torch.softmax(instance_attn_score, 0)
            loss = torch.triu(instance_attn_score_normed.T @ instance_attn_score_normed, diagonal=1).mean()
        elif 'delta_logits' in results_dict:
            delta_logits = results_dict['delta_logits']
            loss = self.loss(delta_logits, label) * 1e-4
        elif 'text_cluster_prob' in results_dict:
            text_cluster_prob = results_dict['text_cluster_prob']
            image_cluster_prob = results_dict['image_cluster_prob']

            loss += F.kl_div(image_cluster_prob.log(), text_cluster_prob, reduction='batchmean')

        # ---->loss
        loss += self.loss(logits, label)

        self.training_step_outputs.append({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label})
        return loss

    def on_train_epoch_end(self):
        current_epoch = self.current_epoch + 1 # Assuming current_epoch is 0-indexed
        training_step_outputs = self.training_step_outputs

        logits = torch.cat([x['logits'] for x in training_step_outputs], dim=0).cpu().detach()
        all_probs = torch.cat([x['Y_prob'] for x in training_step_outputs], dim=0).cpu().detach()
        all_pred = torch.stack([x['Y_hat'].squeeze() for x in training_step_outputs]).cpu().detach()
        all_labels = torch.stack([x['label'] for x in training_step_outputs], dim=0).cpu().detach()

        # Standard Accuracy (optional)
        train_acc_standard = torch.sum(all_pred == all_labels.squeeze()).item() / len(all_labels)

        # Balanced Accuracy
        # Ensure labels and predictions are 1D NumPy arrays
        y_true_np = all_labels.squeeze().numpy()
        y_pred_np = all_pred.numpy()
        train_balanced_acc = balanced_accuracy_score(y_true_np, y_pred_np)

        train_f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        train_precision = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        train_recall = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)

        if self.n_classes == 2:
            # Ensure y_true_np is 1D for roc_auc_score if it's binary
            auc = roc_auc_score(y_true_np, all_probs[:, 1].numpy())
            aucs = [] # Not used in binary case, but kept for consistency
        else:
            aucs = []
            # Ensure y_true_np is 1D for label_binarize
            binary_labels = label_binarize(y_true_np, classes=[i for i in range(self.n_classes)])
            for class_idx in range(self.n_classes):
                # Check if the class is present in true labels
                if np.sum(binary_labels[:, class_idx]) > 0: # Check if class has any true samples
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx].numpy())
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan')) # Or handle as 0 or skip

            auc = np.nanmean(np.array(aucs)) if aucs else float('nan')


        # Beautified output
        print(Fore.CYAN + "═" * 80)
        print(Fore.GREEN + f"【TRAINING RESULTS】- Epoch: {Fore.YELLOW}{current_epoch}{Style.RESET_ALL}")
        print(f"  Accuracy: {Fore.YELLOW}{train_acc_standard:.4f}{Style.RESET_ALL}")
        print(f"  Balanced Accuracy:   {Fore.BLUE}{train_balanced_acc:.4f}{Style.RESET_ALL}") # Added
        print(f"  F1 Score:            {Fore.YELLOW}{train_f1:.4f}{Style.RESET_ALL}")
        print(f"  Precision:           {Fore.YELLOW}{train_precision:.4f}{Style.RESET_ALL}")
        print(f"  Recall:              {Fore.YELLOW}{train_recall:.4f}{Style.RESET_ALL}")
        print(f"  AUC:                 {Fore.YELLOW}{auc:.4f}{Style.RESET_ALL}")
        print(Fore.CYAN + "═" * 80)

        # Optional: Log training metrics (if you have a self.log method like in PyTorch Lightning)
        # self.log('train_acc_standard', train_acc_standard, on_epoch=True, logger=True)
        # self.log('train_balanced_acc', train_balanced_acc, on_epoch=True, logger=True)
        # self.log('train_f1', train_f1, on_epoch=True, logger=True)
        # self.log('train_precision', train_precision, on_epoch=True, logger=True)
        # self.log('train_recall', train_recall, on_epoch=True, logger=True)
        # self.log('train_auc', auc, on_epoch=True, logger=True)

        self.training_step_outputs = [] # Clear the outputs for the next training epoch


    def validation_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']
        self.val_step_outputs.append({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label})

    def on_validation_epoch_end(self):
        current_epoch = self.current_epoch + 1 # Assuming current_epoch is 0-indexed
        val_step_outputs = self.val_step_outputs

        logits = torch.cat([x['logits'] for x in val_step_outputs], dim=0).cpu().detach()
        all_probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim=0).cpu().detach()
        all_pred = torch.stack([x['Y_hat'].squeeze() for x in val_step_outputs]).cpu().detach()
        all_labels = torch.stack([x['label'] for x in val_step_outputs], dim=0).cpu().detach()

        # Standard Accuracy (optional)
        val_acc_standard = torch.sum(all_pred == all_labels.squeeze()).item() / len(all_labels)

        # Balanced Accuracy
        y_true_np = all_labels.squeeze().numpy()
        y_pred_np = all_pred.numpy()
        val_balanced_acc = balanced_accuracy_score(y_true_np, y_pred_np)

        val_f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        val_precision = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        val_recall = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)

        if self.n_classes == 2:
            auc = roc_auc_score(y_true_np, all_probs[:, 1].numpy())
            aucs = [] # Not used in binary case, but kept for consistency
        else:
            aucs = []
            # Ensure y_true_np is 1D for label_binarize
            binary_labels = label_binarize(y_true_np, classes=[i for i in range(self.n_classes)])
            for class_idx in range(self.n_classes):
                # Check if the class is present in true labels
                if np.sum(binary_labels[:, class_idx]) > 0:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx].numpy())
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan')) # Or handle as 0 or skip

            auc = np.nanmean(np.array(aucs)) if aucs else float('nan')


        # Beautified output
        print(Fore.CYAN + "═" * 80)
        print(Fore.MAGENTA + f"【VALIDATION RESULTS】- Epoch: {Fore.YELLOW}{current_epoch}{Style.RESET_ALL}")
        print(f"  Accuracy: {Fore.YELLOW}{val_acc_standard:.4f}{Style.RESET_ALL}")
        print(f"  Balanced Accuracy:   {Fore.GREEN}{val_balanced_acc:.4f}{Style.RESET_ALL}") # Added
        print(f"  F1 Score:            {Fore.YELLOW}{val_f1:.4f}{Style.RESET_ALL}")
        print(f"  Precision:           {Fore.YELLOW}{val_precision:.4f}{Style.RESET_ALL}")
        print(f"  Recall:              {Fore.YELLOW}{val_recall:.4f}{Style.RESET_ALL}")
        print(f"  AUC:                 {Fore.YELLOW}{auc:.4f}{Style.RESET_ALL}")
        print(Fore.CYAN + "═" * 80)

        # Log validation metrics
        self.log('val_f1', val_f1, prog_bar=True, on_epoch=True, logger=True)
        self.log('val_acc', val_acc_standard, prog_bar=True, on_epoch=True, logger=True) # Renamed for clarity
        self.log('val_balanced_acc', val_balanced_acc, prog_bar=True, on_epoch=True, logger=True) # Added
        self.log('val_precision', val_precision, on_epoch=True, logger=True)
        self.log('val_recall', val_recall, on_epoch=True, logger=True)
        self.log('val_auc', auc, on_epoch=True, logger=True)

        self.val_step_outputs = [] # Clear the outputs for the next validation epoch


    def test_step(self, batch, batch_idx):
        data, label = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict['logits']
        Y_prob = results_dict['Y_prob']
        Y_hat = results_dict['Y_hat']

        self.test_step_outputs.append({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'label': label})

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            print(Fore.YELLOW + "No test outputs to process." + Style.RESET_ALL)
            return

        test_step_outputs = self.test_step_outputs
        all_logits = torch.cat([x['logits'] for x in test_step_outputs], dim=0).cpu().detach()
        all_probs = torch.cat([x['Y_prob'] for x in test_step_outputs], dim=0).cpu().detach()
        # Ensure Y_hat is consistently shaped, especially if it comes from different batch sizes
        # Assuming Y_hat might be [batch_size, 1] or [batch_size], we use squeeze() and then stack.
        # If Y_hat is already consistently [batch_size], then stack is fine.
        # Let's adjust to handle potential single-element tensors that become 0-dim after squeeze.
        processed_Y_hat = []
        for x in test_step_outputs:
            y_hat_squeezed = x['Y_hat'].squeeze()
            if y_hat_squeezed.ndim == 0:  # If it became a 0-dim tensor
                y_hat_squeezed = y_hat_squeezed.unsqueeze(0)  # Make it 1-dim
            processed_Y_hat.append(y_hat_squeezed)
        all_pred = torch.cat(processed_Y_hat, dim=0).cpu().detach()

        # Ensure all_labels is consistently shaped. Assuming labels are [batch_size] or [batch_size, 1]
        processed_labels = []
        for x in test_step_outputs:
            label_squeezed = x['label'].squeeze()
            if label_squeezed.ndim == 0:  # If it became a 0-dim tensor
                label_squeezed = label_squeezed.unsqueeze(0)  # Make it 1-dim
            processed_labels.append(label_squeezed)
        all_labels = torch.cat(processed_labels, dim=0).cpu().detach()

        if all_pred.shape != all_labels.shape:
            print(
                Fore.RED + f"Shape mismatch! Predictions shape: {all_pred.shape}, Labels shape: {all_labels.shape}" + Style.RESET_ALL)
            # Attempt to squeeze labels if it's the common case of [N, 1] vs [N]
            if all_labels.ndim > 1 and all_labels.shape[1] == 1:
                all_labels_squeezed = all_labels.squeeze()
                if all_pred.shape == all_labels_squeezed.shape:
                    all_labels = all_labels_squeezed
                    print(Fore.YELLOW + f"Adjusted labels shape to: {all_labels.shape}" + Style.RESET_ALL)
                else:
                    print(
                        Fore.RED + "Could not resolve shape mismatch automatically. Please check your data." + Style.RESET_ALL)
                    return
            else:
                print(
                    Fore.RED + "Could not resolve shape mismatch automatically. Please check your data." + Style.RESET_ALL)
                return

        test_acc = torch.sum(all_pred == all_labels).item() / len(all_labels)

        # Balanced Accuracy
        test_balanced_acc = balanced_accuracy_score(all_labels.numpy(), all_pred.numpy())

        test_f1 = f1_score(all_labels.numpy(), all_pred.numpy(), average='macro', zero_division=0)
        test_precision = precision_score(all_labels.numpy(), all_pred.numpy(), average='macro', zero_division=0)
        test_recall = recall_score(all_labels.numpy(), all_pred.numpy(), average='macro', zero_division=0)

        if self.n_classes == 2:
            auc = roc_auc_score(all_labels.numpy(), all_probs[:, 1].numpy())
            aucs = []
        else:
            aucs = []
            binary_labels = label_binarize(all_labels.numpy(), classes=[i for i in range(self.n_classes)])
            for class_idx in range(self.n_classes):
                if np.sum(binary_labels[:, class_idx]) > 0 and binary_labels.shape[
                    1] > class_idx:  # Check if class is present and index is valid
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx].numpy())
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))

            auc = np.nanmean(np.array(aucs)) if aucs else float('nan')

        # --- 计算并打印每个类别的准确率 ---
        print(Fore.CYAN + "─" * 30 + " PER-CLASS ACCURACY " + "─" * 30 + Style.RESET_ALL)
        unique_labels = torch.unique(all_labels)
        class_accuracies = {}

        for label_class in unique_labels:
            class_idx = (all_labels == label_class)
            class_preds = all_pred[class_idx]
            class_true_labels = all_labels[class_idx]

            correct_predictions = torch.sum(class_preds == class_true_labels).item()
            total_instances = len(class_true_labels)

            if total_instances > 0:
                accuracy_for_class = correct_predictions / total_instances
                class_accuracies[label_class.item()] = accuracy_for_class
                print(
                    f"  Accuracy for Class {label_class.item()}: {Fore.MAGENTA}{accuracy_for_class:.4f}{Style.RESET_ALL} ({correct_predictions}/{total_instances})")
            else:
                class_accuracies[label_class.item()] = float('nan')  # 或者 0，根据你的偏好
                print(
                    f"  Accuracy for Class {label_class.item()}: {Fore.MAGENTA}N/A{Style.RESET_ALL} (No instances found)")
        # --- 结束每个类别准确率的计算和打印 ---

        print(Fore.CYAN + "═" * 80)
        print(Fore.RED + "【TEST RESULTS】" + Style.RESET_ALL)
        print(f"  Accuracy:    {Fore.YELLOW}{test_acc:.4f}{Style.RESET_ALL}")
        print(f"  Balanced Accuracy:   {Fore.GREEN}{test_balanced_acc:.4f}{Style.RESET_ALL}")
        print(f"  F1 Score:    {Fore.YELLOW}{test_f1:.4f}{Style.RESET_ALL}")
        print(f"  Precision:   {Fore.YELLOW}{test_precision:.4f}{Style.RESET_ALL}")
        print(f"  Recall:      {Fore.YELLOW}{test_recall:.4f}{Style.RESET_ALL}")
        print(f"  AUC:     {Fore.YELLOW}{auc:.4f}{Style.RESET_ALL}")
        print(
            f"  Data Shapes: {Fore.BLUE}logits {all_logits.shape}, labels {all_labels.shape}, predictions {all_pred.shape}{Style.RESET_ALL}")
        print(Fore.CYAN + "═" * 80)

        print(Fore.GREEN + "Saving results to: " + Style.RESET_ALL + f"{self.log_path}")

        metrics = {
            'acc': [test_acc],
            'balanced_acc': [test_balanced_acc],
            'f1': [test_f1],
            'precision': [test_precision],
            'recall': [test_recall],
            'auc': [auc]
        }
        # 添加每个类别的准确率到 metrics 字典
        for class_val, acc in class_accuracies.items():
            metrics[f'class_{class_val}_acc'] = [acc]

        df = pd.DataFrame(metrics)
        df.to_csv(os.path.join(self.log_path, 'test_metrics.csv'), index=False)

        probs_df = pd.DataFrame(all_probs.numpy())
        probs_df.to_csv(os.path.join(self.log_path, 'all_probs.csv'), index=False)

        labels_df = pd.DataFrame(all_labels.numpy())
        labels_df.to_csv(os.path.join(self.log_path, 'all_labels.csv'), index=False)

        logits_df = pd.DataFrame(all_logits.numpy())
        logits_df.to_csv(os.path.join(self.log_path, 'all_logits.csv'), index=False)

        np.savez_compressed(os.path.join(self.log_path, 'all_probs.npz'), all_probs.numpy())
        np.savez_compressed(os.path.join(self.log_path, 'all_labels.npz'), all_labels.numpy())
        np.savez_compressed(os.path.join(self.log_path, 'all_logits.npz'), all_logits.numpy())

        self.test_step_outputs = []

    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]

    def load_model(self):
        name = self.hparams.model.name
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
            print(Fore.GREEN + f"Model loaded successfully: {Fore.CYAN}{camel_name}")
        except:
            raise ValueError(Fore.RED + 'Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)