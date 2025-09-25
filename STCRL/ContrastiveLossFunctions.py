import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import math
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class TemporalContrastiveLoss(nn.Module):
    """Base class for temporal contrastive losses"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

class CompletionTimeLoss(TemporalContrastiveLoss):
    """Contrastive loss using completion time similarity"""
    def forward(self, projections, temporal_data):
        completion_times = temporal_data['completion_time']

        proj_norm = F.normalize(projections, dim=1)
        sim_matrix = torch.mm(proj_norm, proj_norm.t()) / self.temperature

        time_diffs = torch.abs(completion_times.unsqueeze(0) - completion_times.unsqueeze(1))
        time_sim = 1 - (time_diffs / (time_diffs.max() + 1e-6))

        weights = time_sim.clone()
        weights.fill_diagonal_(0)

        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)
        loss = -(weights * log_prob).sum() / (weights.sum() + 1e-6)

        return loss

class TaskTypeLoss(TemporalContrastiveLoss):
    """Contrastive loss based on task type (unimanual vs bimanual)"""
    def forward(self, projections, temporal_data):
        task_types = temporal_data['task_type']

        proj_norm = F.normalize(projections, dim=1)
        sim_matrix = torch.mm(proj_norm, proj_norm.t()) / self.temperature

        # Create binary similarity matrix (1 for same task type, 0 for different)
        task_sim = 1 - torch.abs(task_types.unsqueeze(0) - task_types.unsqueeze(1))

        weights = task_sim.clone()
        weights.fill_diagonal_(0)

        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)
        loss = -(weights * log_prob).sum() / (weights.sum() + 1e-6)

        return loss

class RMSDLoss(TemporalContrastiveLoss):
    """Contrastive loss based on RMSD profiles"""
    def forward(self, projections, temporal_data):
        rmsd_values = temporal_data['rmsd']
        completion_times = temporal_data['completion_time']

        proj_norm = F.normalize(projections, dim=1)
        sim_matrix = torch.mm(proj_norm, proj_norm.t()) / self.temperature

        # Normalize RMSD by completion time for rate
        # Handle potential zero completion_time values
        safe_completion_times = torch.max(completion_times, torch.ones_like(completion_times) * 1e-6)
        rmsd_rates = rmsd_values / safe_completion_times
        rmsd_diffs = torch.abs(rmsd_rates.unsqueeze(0) - rmsd_rates.unsqueeze(1))
        rmsd_sim = 1 - (rmsd_diffs / (rmsd_diffs.max() + 1e-6))

        weights = rmsd_sim.clone()
        weights.fill_diagonal_(0)

        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)
        loss = -(weights * log_prob).sum() / (weights.sum() + 1e-6)

        return loss

class SuccessLoss(TemporalContrastiveLoss):
    """Contrastive loss based on task success"""
    def forward(self, projections, temporal_data):
        success_values = temporal_data['is_success']

        proj_norm = F.normalize(projections, dim=1)
        sim_matrix = torch.mm(proj_norm, proj_norm.t()) / self.temperature

        # Create binary similarity matrix (1 for same success status, 0 for different)
        success_sim = 1 - torch.abs(success_values.unsqueeze(0) - success_values.unsqueeze(1))

        weights = success_sim.clone()
        weights.fill_diagonal_(0)

        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)
        loss = -(weights * log_prob).sum() / (weights.sum() + 1e-6)

        return loss

class WithinBetweenSubjectLoss(TemporalContrastiveLoss):
    """Contrastive loss that balances within-subject and between-subject learning"""
    def __init__(self, temperature=0.07, within_weight=0.7):
        super().__init__(temperature)
        self.within_weight = within_weight

    def forward(self, projections, temporal_data):
        participant_ids = temporal_data['participant_id']

        batch_size = projections.shape[0]
        proj_norm = F.normalize(projections, dim=1)
        sim_matrix = torch.mm(proj_norm, proj_norm.t()) / self.temperature

        # Create participant similarity matrix (1 for same participant, 0 for different)
        same_participant = (participant_ids.unsqueeze(0) == participant_ids.unsqueeze(1)).float()

        # Remove self-similarity
        mask_self = torch.eye(batch_size, device=projections.device)
        same_participant = same_participant * (1 - mask_self)
        diff_participant = (1 - same_participant) * (1 - mask_self)

        # Combine within and between subject matrices with weighting
        combined_weights = self.within_weight * same_participant + (1 - self.within_weight) * diff_participant

        # Calculate loss
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)
        loss = -(combined_weights * log_prob).sum() / (combined_weights.sum() + 1e-6)

        return loss

class MultiTemporalLoss(TemporalContrastiveLoss):
    """Combines multiple temporal loss functions"""
    def __init__(self, temperature=0.07, weights=(0.2, 0.2, 0.2, 0.2, 0.2), within_subject_weight=0.7):
        super().__init__(temperature)
        self.completion_loss = CompletionTimeLoss(temperature)
        self.task_type_loss = TaskTypeLoss(temperature)
        self.rmsd_loss = RMSDLoss(temperature)
        self.success_loss = SuccessLoss(temperature)
        self.subject_loss = WithinBetweenSubjectLoss(temperature, within_subject_weight)
        self.weights = weights
        self.within_subject_weight = within_subject_weight

    def forward(self, projections, temporal_data):
        loss1 = self.completion_loss(projections, temporal_data)
        loss2 = self.task_type_loss(projections, temporal_data)
        loss3 = self.rmsd_loss(projections, temporal_data)
        loss4 = self.success_loss(projections, temporal_data)
        loss5 = self.subject_loss(projections, temporal_data)

        return (self.weights[0] * loss1 +
                self.weights[1] * loss2 +
                self.weights[2] * loss3 +
                self.weights[3] * loss4 +
                self.weights[4] * loss5)