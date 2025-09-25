import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import copy
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union


class AdaptiveLossWeighter:
    """
    Advanced adaptive loss weighting system for cross-task and cross-subject transfer learning.
    Adjusts alpha parameters based on transfer paradigms.
    """

    def __init__(self, num_losses=5, initial_weights=None,
                 history_window=10, min_weight=0.05, adapt_speed=0.1,
                 transfer_type='cross_subject'):
        """
        Args:
            transfer_type: 'cross_subject', 'cross_task', or 'cross_both'
        """
        # Initialize weights based on transfer type
        if initial_weights is None:
            if transfer_type == 'cross_subject':
                # For cross-subject: emphasize completion time and within-subject similarities
                initial_weights = [0.3, 0.1, 0.2, 0.2, 0.2]
            elif transfer_type == 'cross_task':
                # For cross-task: emphasize task type and RMSD patterns
                initial_weights = [0.2, 0.3, 0.3, 0.1, 0.1]
            else:  # cross_both
                # Balanced approach
                initial_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

        self.transfer_type = transfer_type
        self.weights = torch.tensor(initial_weights, dtype=torch.float32)
        self.alpha_params = torch.tensor(initial_weights, dtype=torch.float32)  # Transfer-specific alphas
        self.loss_history = [[] for _ in range(len(initial_weights))]
        self.loss_improvement = [0.0] * len(initial_weights)
        self.grad_norms = [[] for _ in range(len(initial_weights))]
        self.history_window = history_window
        self.min_weight = min_weight
        self.adapt_speed = adapt_speed
        self.iterations = 0
        self.transfer_phase = 'initial'  # 'initial', 'fine_tune', 'zero_shot'

    def set_transfer_phase(self, phase: str):
        """Set the current transfer learning phase"""
        self.transfer_phase = phase

        # Adjust adaptation behavior based on phase
        if phase == 'zero_shot':
            # No adaptation in zero-shot
            self.adapt_speed = 0.0
        elif phase == 'fine_tune':
            # Faster adaptation during fine-tuning
            self.adapt_speed = 0.15
        else:
            # Standard adaptation
            self.adapt_speed = 0.1

    def get_alpha_adjusted_weights(self, domain_similarity=0.5):
        """
        Get alpha-adjusted weights based on transfer paradigm and domain similarity.

        Args:
            domain_similarity: Float between 0-1 indicating similarity between source and target domains
        """
        base_weights = self.weights.clone()

        # Adjust alpha parameters based on transfer type and domain similarity
        if self.transfer_type == 'cross_subject':
            # For cross-subject transfer, reduce within-subject weight, increase completion time weight
            alpha_adjustments = torch.tensor([
                1.0 + 0.3 * (1 - domain_similarity),  # completion_time
                1.0 - 0.2 * (1 - domain_similarity),  # task_type
                1.0,  # rmsd
                1.0,  # success
                1.0 - 0.5 * (1 - domain_similarity)  # within_subject (reduce for new subjects)
            ])
        elif self.transfer_type == 'cross_task':
            # For cross-task transfer, emphasize task-specific features
            alpha_adjustments = torch.tensor([
                1.0 - 0.2 * (1 - domain_similarity),  # completion_time
                1.0 + 0.4 * (1 - domain_similarity),  # task_type (increase for new tasks)
                1.0 + 0.3 * (1 - domain_similarity),  # rmsd (increase for new movement patterns)
                1.0 + 0.1 * (1 - domain_similarity),  # success
                1.0  # within_subject
            ])
        else:  # cross_both
            # Balanced adjustment for both types of transfer
            alpha_adjustments = torch.tensor([
                1.0 + 0.1 * (1 - domain_similarity),
                1.0 + 0.2 * (1 - domain_similarity),
                1.0 + 0.2 * (1 - domain_similarity),
                1.0 + 0.1 * (1 - domain_similarity),
                1.0 - 0.3 * (1 - domain_similarity)
            ])

        # Apply alpha adjustments
        adjusted_weights = base_weights * alpha_adjustments

        # Normalize
        adjusted_weights = adjusted_weights / adjusted_weights.sum()

        # Ensure minimum weights
        adjusted_weights = torch.clamp(adjusted_weights, min=self.min_weight)
        adjusted_weights = adjusted_weights / adjusted_weights.sum()

        return adjusted_weights

    def update_history(self, current_losses, grad_norms=None):
        """Update loss history and calculate improvement rates"""
        for i, loss in enumerate(current_losses):
            self.loss_history[i].append(loss.item() if torch.is_tensor(loss) else loss)

            if len(self.loss_history[i]) > self.history_window:
                self.loss_history[i].pop(0)

            if grad_norms is not None and i < len(grad_norms):
                self.grad_norms[i].append(grad_norms[i])
                if len(self.grad_norms[i]) > self.history_window:
                    self.grad_norms[i].pop(0)

        self.iterations += 1

    def get_updated_weights(self, domain_similarity=0.5):
        """Get updated weights with alpha adjustments"""
        # First get standard adaptive weights
        if self.iterations >= self.history_window and self.transfer_phase != 'zero_shot':
            # Calculate improvement-based weights
            improvement_rates = []
            for i, history in enumerate(self.loss_history):
                if len(history) >= self.history_window:
                    mid_point = len(history) // 2
                    earlier_avg = sum(history[:mid_point]) / mid_point
                    later_avg = sum(history[mid_point:]) / (len(history) - mid_point)

                    if later_avg > 0:
                        improvement = earlier_avg / later_avg
                        improvement_rates.append(improvement)
                    else:
                        improvement_rates.append(1.0)
                else:
                    improvement_rates.append(1.0)

            # Convert to weights (inverse of improvement - less improving losses get higher weights)
            improvement_weights = torch.tensor([1.0 / (rate + 0.5) for rate in improvement_rates])
            improvement_weights = improvement_weights / improvement_weights.sum()

            # Combine with current weights
            self.weights = (1 - self.adapt_speed) * self.weights + self.adapt_speed * improvement_weights

        # Apply alpha adjustments based on transfer paradigm
        final_weights = self.get_alpha_adjusted_weights(domain_similarity)

        return final_weights
