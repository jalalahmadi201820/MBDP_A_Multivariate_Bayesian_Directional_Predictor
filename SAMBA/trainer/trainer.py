# -*- coding: utf-8 -*-
"""
Trainer class for SAMBA model
"""

import os
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from utils.logger import get_logger
from utils.metrics import All_Metrics


class Trainer:
    """Trainer class for SAMBA model"""
    
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        
        if val_loader is not None:
            self.val_per_epoch = len(val_loader)
        
        self.best_path = os.path.join(self.args.get('log_dir'), 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.get('log_dir'), 'loss.png')
        
        # Setup logging
        if os.path.isdir(args.get('log_dir')) == False and not args.get('debug'):
            os.makedirs(args.get('log_dir'), exist_ok=True)
        self.logger = get_logger(args.get('log_dir'), name=args.get('model'), debug=args.get('debug'))
        self.logger.info('Experiment log path in: {}'.format(args.get('log_dir')))
    
    def val_epoch(self, epoch, val_dataloader):
        """Validation epoch"""
        self.model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data
                label = target
                output = self.model(data)
                loss = self.loss(output, label)
                
                # A whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        
        val_loss = total_val_loss / len(val_dataloader)
        self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss
    
    def train_epoch(self, epoch):
        """Training epoch"""
        self.model.train()
        total_loss = 0
        loss_values = []
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data
            label = target  # (..., 1)
            self.optimizer.zero_grad()
            
            # Data and target shape: B, T, N, F; output shape: B, T, N, F
            output = self.model(data)
            loss = self.loss(output, label)
            loss.backward()
            
            # Add max grad clipping
            if self.args.get('grad_norm'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.get('max_grad_norm'))
            
            self.optimizer.step()
            total_loss += loss.item()
            loss_values.append(loss.item())
            
            # Log information
            if batch_idx % self.args.get('log_step') == 0:
                self.logger.info('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        
        train_epoch_loss = total_loss / self.train_per_epoch
        self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, train_epoch_loss))
        
        # Learning rate decay
        if self.args.get('lr_decay'):
            self.lr_scheduler.step()
        
        return train_epoch_loss
    
    def train(self):
        """Main training loop"""
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        
        for epoch in range(1, self.args.get('epochs') + 1):
            train_epoch_loss = self.train_epoch(epoch)
            
            if self.val_loader is None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)
            
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            
            # Early stop
            if self.args.get('early_stop'):
                if not_improved_count == self.args.get('early_stop_patience'):
                    self.logger.info("Validation performance didn't improve for {} epochs. "
                                     "Training stops.".format(self.args.get('early_stop_patience')))
                    break
            
            # Save the best state
            if best_state:
                self.logger.info('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
        
        training_time = time.time() - start_time
        self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        
        # Save the best model to file
        if not self.args.get('debug'):
            torch.save(best_model, self.best_path)
            self.logger.info("Saving current best model to " + self.best_path)
        
        # Test
        self.model.load_state_dict(best_model)
        y1, y2 = self.test(self.model, self.args, self.test_loader, self.logger)
        
        return y1, y2
    
    def save_checkpoint(self):
        """Save training checkpoint"""
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)
    
    @staticmethod
    def test(model, args, data_loader, logger, path=None):
        """Test the model"""
        if path is not None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.get('device'))
        
        model.eval()
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data
                label = target
                output = model(data)
                
                y_true.append(label)
                y_pred.append(output)
        
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        
        mae, rmse, _ = All_Metrics(y_pred, y_true, args.get('mae_thresh'), args.get('mape_thresh'))
        logger.info("Average Horizon, MAE: {:.4f}, MSE: {:.4f}".format(mae, rmse))
        
        return y_pred, y_true
    
    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """Compute the sampling probability for scheduled sampling using inverse sigmoid."""
        import math
        return k / (k + math.exp(global_step / k))
