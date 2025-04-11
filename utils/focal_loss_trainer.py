import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

class FocalLossMultiClass(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', loss_scale=1.0):
        super(FocalLossMultiClass, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_scale = loss_scale

    def forward(self, inputs, targets):
        device = inputs.device  # Ensure all tensors go to the same device

        inputs = inputs.view(-1, inputs.size(-1))  # [N, C]
        targets = targets.view(-1)                 # [N]

        mask = targets != -100
        inputs = inputs[mask]
        targets = targets[mask]

        targets = targets.to(device)  # ✅ Ensure targets are on the same device
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)

        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1)).float().to(device)  # ✅ to same device

        pt = (probs * targets_one_hot).sum(dim=-1)
        log_pt = (log_probs * targets_one_hot).sum(dim=-1)

        if self.alpha is not None:
            alpha = self.alpha.to(device)  # ✅ Move class weights to same device
            at = alpha[targets]
            loss = -at * ((1 - pt) ** self.gamma) * log_pt
        else:
            loss = -((1 - pt) ** self.gamma) * log_pt

        if self.reduction == 'mean':
            return loss.mean() * self.loss_scale
        elif self.reduction == 'sum':
            return loss.sum() * self.loss_scale
        else:
            return loss * self.loss_scale


class FocalLossTrainer(Trainer):
    def __init__(self, *args, alpha, gamma, loss_scale, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLossMultiClass(alpha=alpha, gamma=gamma, loss_scale=loss_scale)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")  # Remove labels for manual loss calculation
        outputs = model(**inputs)
        logits = outputs.logits

        loss = self.focal_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss
