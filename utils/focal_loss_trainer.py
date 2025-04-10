import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

class FocalLossMultiClass(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLossMultiClass, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.view(-1, inputs.size(-1))  # [N, C]
        targets = targets.view(-1)                 # [N]

        mask = targets != -100
        inputs = inputs[mask]
        targets = targets[mask]

        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1)).float()

        pt = (probs * targets_one_hot).sum(dim=-1)
        log_pt = (log_probs * targets_one_hot).sum(dim=-1)

        if self.alpha is not None:
            targets = targets.to(self.alpha.device)
            at = self.alpha[targets]
            loss = -at * ((1 - pt) ** self.gamma) * log_pt
        else:
            loss = -((1 - pt) ** self.gamma) * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class FocalLossTrainer(Trainer):
    def __init__(self, *args, alpha, gamma, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLossMultiClass(alpha=alpha, gamma=gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # Remove labels for manual loss calculation
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.focal_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss
