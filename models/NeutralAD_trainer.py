# Neural Transformation Learning for Anomaly Detection (NeuTraLAD) - a self-supervised method for anomaly detection
# Copyright (c) 2022 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import time
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np
from utils import compute_pre_recall_f1,format_time
class NeutralAD_trainer:

    def __init__(self, model, loss_function,criterion,device='cuda',lbd=0.95):

        self.loss_fun = loss_function
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.criterion = criterion
        self.lbd = lbd

    def _train(self,train_loader, optimizer):

        self.model.train()

        loss_all = 0
        for data in train_loader:
            try:
                samples, _, _ = data
            except:
                samples = data

            z, shift = self.model(samples)
            sh_labels = torch.ones(shift.shape[0])
            sh_labels[0:(z.shape[0]*z.shape[1])] = 0
            # K_trans = shift.shape[0]//z.shape[0]
            loss_shift = self.criterion(shift,sh_labels.to(self.device).long()).mean()#.view(z.shape[0],K_trans).sum(1)
            # scale = 1 / np.abs(K_trans * np.log(1.0 / K_trans))
            loss = self.loss_fun(z) 
            total_loss = self.lbd*loss.mean() + (1-self.lbd)*loss_shift
            #loss_mean = total_loss.mean()
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_all += total_loss#.sum()

        return loss_all.item()/len(train_loader)


    def detect_outliers(self, loader,cls):
        model = self.model
        model.eval()

        loss_in = 0
        loss_out = 0
        target_all = []
        score_all = []
        for data in loader:
            with torch.no_grad():
                try:
                    samples, labels, _= data
                except:
                    samples = data
                    labels = data.y!=cls
                z,shift= model(samples)
                score = self.loss_fun(z,eval=True)
                sh_labels = torch.ones(shift.shape[0])
                sh_labels[z.shape[0]:z.shape[1]] = 0
                sh_labels[0:z.shape[0]] = labels
                K_trans = shift.shape[0]//z.shape[0]
                shift_score = F.cross_entropy(shift,sh_labels.to(self.device).long(),reduction='mean')#.view(z.shape[0],K_trans).mean(1)
                # scale = 1 / np.abs(K_trans * np.log(1.0 / K_trans))
                total_score = self.lbd*score + (1-self.lbd)*shift_score
                # print(shift_score)
                # print(score)
                # print(total_score)
                
                loss_in += total_score[labels == 0].sum()
                loss_out += total_score[labels == 1].sum()
                target_all.append(labels)
                score_all.append(total_score)

        try:
            score_all = np.concatenate(score_all)
        except:
            score_all = torch.cat(score_all).cpu().numpy()
        target_all = np.concatenate(target_all)
        auc = roc_auc_score(target_all, score_all)
        f1 = compute_pre_recall_f1(target_all,score_all)
        ap = average_precision_score(target_all, score_all)
        return auc, ap,f1,loss_in.item() / (target_all == 0).sum(), loss_out.item() / (target_all == 1).sum(),score_all,target_all


    def train(self, train_loader,cls = None,max_epochs=100, optimizer=None, scheduler=None,
              validation_loader=None, test_loader=None, early_stopping=None, logger=None, log_every=2):

        early_stopper = early_stopping() if early_stopping is not None else None

        val_auc, val_f1, = -1, -1
        test_auc, test_f1, test_score = None, None,None
        score,target = None,None

        time_per_epoch = []

        for epoch in range(1, max_epochs+1):

            start = time.time()
            train_loss = self._train(train_loader, optimizer)
            end = time.time() - start
            time_per_epoch.append(end)

            if scheduler is not None:
                scheduler.step()

            if test_loader is not None:
                test_auc, test_ap,test_f1, testin_loss,testout_loss,score,target = self.detect_outliers(test_loader,cls)

            if validation_loader is not None:
                val_auc, val_ap,val_f1, valin_loss,valout_loss,_,_ = self.detect_outliers(validation_loader,cls)
                if epoch>5:
                    if early_stopper is not None and early_stopper.stop(epoch, valin_loss, val_auc, testin_loss, test_auc, test_ap,test_f1,
                                                                        train_loss,score,target):
                        break

            if epoch % log_every == 0 or epoch == 1:
                msg = f'Epoch: {epoch}, TR loss: {train_loss}, VAL loss: {valin_loss,valout_loss}, VL auc: {val_auc} VL ap: {val_ap} VL f1: {val_f1} '

                if logger is not None:
                    logger.log(msg)
                    print(msg)
                else:
                    print(msg)

        if early_stopper is not None:
            train_loss, val_loss, val_auc, test_loss, test_auc, test_ap, test_f1, best_epoch,score,target \
                = early_stopper.get_best_vl_metrics()
            msg = f'Stopping at epoch {best_epoch}, TR loss: {train_loss}, VAL loss: {val_loss}, VAL auc: {val_auc} ,' \
                f'TS loss: {test_loss}, TS auc: {test_auc} TS ap: {test_ap} TS f1: {test_f1}'
            if logger is not None:
                logger.log(msg)
                print(msg)
            else:
                print(msg)

        time_per_epoch = torch.tensor(time_per_epoch)
        avg_time_per_epoch = float(time_per_epoch.mean())
        elapsed = format_time(avg_time_per_epoch)

        return val_loss, val_auc, test_auc, test_ap,test_f1,score,target