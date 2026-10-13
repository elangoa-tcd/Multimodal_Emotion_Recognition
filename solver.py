import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
import torch
import torch.nn as nn
# import MultimodalSA.network as new_models
import network as new_models
# import MultimodalSA.Jack_model as new_models
from utils import DiffLoss,SimilarityKL
from utils import to_gpu
        

class Solver(object):
    def __init__(self, train_config, train_data_loader, dev_data_loader, test_data_loader,
                 is_train=True, model=None):
        
        # ---- where this run writes its checkpoints and its log ----
        self.checkpoint_path = train_config.checkpoint_dir
        os.makedirs(os.path.join(self.checkpoint_path, 'devBest'), exist_ok=True)
        self.log_file = os.path.join(self.checkpoint_path, f'log_{train_config.name}.txt')
        with open(self.log_file, "a") as file:
            file.write(f"Log File Time{train_config.name} in {train_config.data}\n")

        self.loss_diff = DiffLoss()
        self.loss_simi = SimilarityKL()

        if train_config.task == 'emotion':
            # pos_weight[i] = (#absent / #present) for emotion i, computed from the TRAIN set.
            # 'full' up-weights rare emotions (surprise, fear); 'sqrt' is a milder version; 'none' = plain BCE.
            _tl = np.array([np.array(s[1]).reshape(-1)[1:7]
                            for s in train_data_loader.dataset.data])
            _present = (_tl > 0).sum(0)
            _pos_weight = torch.tensor((_tl.shape[0] - _present) / np.maximum(_present, 1),
                                       dtype=torch.float32).cuda()
            if train_config.pos_weight == 'sqrt':
                _pos_weight = torch.sqrt(_pos_weight)
            if train_config.pos_weight == 'none':
                print("pos_weight mode: none (unweighted BCE)")
                self.loss_task = nn.BCEWithLogitsLoss(reduction="mean")
            else:
                print(f"pos_weight mode: {train_config.pos_weight}")
                print("pos_weight per emotion [happy,sad,anger,surprise,disgust,fear]:", _pos_weight.tolist())
                self.loss_task = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=_pos_weight)
        else:
            # sentiment: regression on the [-3, 3] score, as in the original TMBL
            self.loss_task = nn.MSELoss(reduction="mean")


        self.train_config = train_config
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model

    def _best_path(self, kind):
        """kind = 'model' or 'optim' -> checkpoints/devBest/<kind>_<run name>.std"""
        return os.path.join(self.checkpoint_path, 'devBest', f'{kind}_{self.train_config.name}.std')
    
    
    def loss_function(self, y_tilde, y):
        """
        total_loss = w1 * cls_loss + w_2 * polar_loss + w_3 * scale_loss
        """
        loss_diff = self.get_diff_loss()
        loss_simi = self.get_simi_loss()
        if self.train_config.task == 'emotion':
            loss_task = self.loss_task(y_tilde, (y > 0).float())   # intensity > 0 -> "present"
        else:
            loss_task = self.loss_task(y_tilde, y)                  # regress the sentiment score directly

        loss = 0.5*loss_diff + 0.05*loss_simi + loss_task
        return loss

    def get_simi_loss(self,):

        # # losses between shared states CMD Original
        # loss = self.loss_simi(self.model.share_T, self.model.share_V, 5) # 使用CMD 需要加上 , 5
        # loss += self.loss_simi(self.model.share_T, self.model.share_A, 5)
        # loss += self.loss_simi(self.model.share_A, self.model.share_V, 5) 
        # loss = loss/3.0

        # SimilarityKL Redesign 
        input = (self.model.share_T, self.model.share_V, self.model.share_A)
        loss = self.loss_simi(input)
        return loss
    
    def get_diff_loss(self):

        shared_t = self.model.share_T
        shared_v = self.model.share_V
        shared_a = self.model.share_A
        private_t = self.model.private_t
        private_v = self.model.private_v
        private_a = self.model.private_a

        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)

        # Across privates
        loss += self.loss_diff(private_a, private_t)
        loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_t, private_v)

        return loss

    def train(self):
        #============================== 
        # Jack Add CheckPoint
        #==============================
        start_epoch = -1 # 手动指定从第几个epoch开始训练
        checkpoint_file = os.path.join(self.checkpoint_path, f"epoch{start_epoch}.pt")
        if self.checkpoint_path is not None and os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
        print(f"=======Start training from epoch {start_epoch}==================")
        curr_patience = patience = self.train_config.patience # 6
        num_trials = self.train_config.trials # 3
 
        best_valid_loss = float('inf')
        best_valid_f1 = -1.0

        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.train_config.milestones, gamma=0.1)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)

        train_losses = []
        train_acces = []
        for e in range(start_epoch,self.train_config.n_epoch): # 0-100

            # === 训练
            self.model.train()
            train_loss = []
            train_acc = []
            for batch in self.train_data_loader:
                y_tilde, y = self.model_input2output(batch)
                loss = self.loss_function(y_tilde, y)
                # acc = self.calc_metrics(y_tilde.cpu().detach().numpy(), y.cpu().detach().numpy(),mode="train") # MD 害群之马 pos/neg

                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
                # train_acc.append(acc)

            train_losses.append(train_loss)
            # train_acces.append(train_acc)
            # print(f"Epoch {e} - Training acc: {round(np.mean(train_acc), 4)}")
            from datetime import datetime
            time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')            
            print(f"\nEpoch {e} - Training loss: {round(np.mean(train_loss), 4)}  Time: {time_now}\n")
            with open(self.log_file, "a") as file:
                # file.write(f"Epoch {e} - Training acc: {round(np.mean(train_acc), 4)} Time: {time_now}\n")            
                file.write(f"\nEpoch {e} - Training loss: {round(np.mean(train_loss), 4)}  Time: {time_now}\n")

            # ==== 验证
            valid_loss, valid_acc = self.eval(mode="dev")
            print(f"val loss: {round(np.mean(valid_loss), 4)}, val acc: {round(np.mean(valid_acc), 4)}")
            with open(self.log_file, "a") as file:
                file.write(f"val loss: {round(np.mean(valid_loss), 4)}, val acc: {round(np.mean(valid_acc), 4)}\n")            
            
            # ==== 测试
            if e % self.train_config.test_duration == 0: # 1 个 epoch 测试一次
                # lr_scheduler.step()
                test_loss, test_acc = self.eval(e,mode="test")
                print(f"test loss: {round(np.mean(test_loss), 4)}, test acc: {round(np.mean(test_acc), 4)}")
                with open(self.log_file, "a") as file:
                    file.write(f"test loss: {round(np.mean(test_loss), 4)}, test acc: {round(np.mean(test_acc), 4)}\n")            

            # ===== 交叉验证(正则化): 若模型一直没有提升，则只允许更新 6*3=18 epoch self.train_config.patience * self.train_config.trials
            # if valid_loss <= best_valid_loss:
            #     best_valid_loss = valid_loss
            if valid_acc >= best_valid_f1:
                best_valid_f1 = valid_acc
                print("Found new best model on dev set!")
                with open(self.log_file, "a") as file:
                    file.write(f"Found new best model on dev set!\n")            
                torch.save(self.model.state_dict(), self.best_path('model'))
                torch.save(self.optimizer.state_dict(),	self._best_path('optim'))
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    with open(self.log_file, "a") as file:
                        file.write(f"Running out of patience, loading previous best model.")  
                    num_trials -= 1
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(self._best_path('model')))
                    self.optimizer.load_state_dict(torch.load(	self._best_path('optim')))
                    lr_scheduler.step()
                    print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
                    with open(self.log_file, "a") as file:
                        file.write(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")              
            file.close()

            # ====== 定点保存
            if e % 10 == 0: # 10 个 epoch 保存一次
                self.model.eval() # 将模型设置为评估模式，这样可以确保所有的参数都被正确保存 0524
                torch.save({
                    'epoch': e,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                },os.path.join(self.checkpoint_path, f"epoch{e}.pt"))
                test_loss, test_acc = self.eval(e,mode="test", to_print=True)
            
            print("#" * 100)
            with open(self.log_file, "a") as file:
                file.write("#" * 100)       

            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break            

        # 最终测试
        self.eval(self.train_config.n_epoch-10,mode="test", to_print=True)

    def eval(self, e=None, mode=None, to_print=False):
        assert (mode is not None)
        self.model.eval()

        y_true, y_pred, y2_pred = [], [], []
        eval_loss, eval_loss_diff = [], []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader
            if to_print:
                self.model.load_state_dict(torch.load(self._best_path('model')))
                print("================== Start Testing ====================================")

        with torch.no_grad():
            for batch in dataloader:
                y_tilde, y = self.model_input2output(batch)
                loss = self.loss_function(y_tilde, y)

                eval_loss.append(loss.item())
                y_pred.append(y_tilde.detach().cpu().numpy())

                y_true.append(y.detach().cpu().numpy())

        eval_loss = np.mean(eval_loss)

        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()

        accuracy = self.calc_metrics(y_true, y_pred, mode, to_print)

        return eval_loss, accuracy

    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

    # def calc_metrics(self, y_true, y_pred, mode=None, to_print=False): # Jack Change 反了
    #     """
    #     Metric scheme adapted from:
    #     https://github.com/yaohungt/Multimodal-Transformer/blob/master/src/eval_metrics.py
    #     """
    #     test_preds = y_pred
    #     test_truth = y_true
    #     non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

    #     test_preds_a7 = np.clip(test_preds, a_min=-3, a_max=3.)
    #     test_truth_a7 = np.clip(test_truth, a_min=-3, a_max=3.)
    #     test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    #     test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    #     mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
    #     corr = np.corrcoef(test_preds, test_truth)[0][1]
    #     mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)
    #     mult_a5 = self.multiclass_acc(test_preds_a5, test_truth_a5)

    #     # ============ pos - neg
    #     binary_truth = (test_truth[non_zeros] > 0)
    #     binary_preds = (test_preds[non_zeros] > 0)

    #     if mode=="train":
    #         pass
    #     if mode=="dev":
    #         print("val_mult_acc7: ", mult_a7)
    #         with open(log_file, "a") as file:
    #                 file.write(f"\nval_mult_acc7 {mult_a7}\n")
    #     f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')
    #     if mode=="test":
    #         print("\ntest_mult_acc7: ", mult_a7)
    #         print("mae: ", mae)
    #         print("corr: ", corr)
    #         print("F1 (pos/neg) :", f_score)
    #         with open(log_file, "a") as file:
    #                 file.write(f"\ntest_mult_acc7 {mult_a7}\n")
    #                 file.write(f"mae: {mae}\n")
    #                 file.write(f"corr: {corr}\n")
    #                 file.write(f"F1 (pos/neg) : {f_score}\n")
    #     f_score = f1_score((test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average='weighted')

    #     if to_print:
    #         print("mae: ", mae)
    #         print("corr: ", corr)
    #         print("mult_acc5: ", mult_a5)
    #         print("mult_acc7: \n", mult_a7)
    #         print("Classification Report (pos/neg) \n:")
    #         print(classification_report(binary_truth, binary_preds, digits=5))
    #         print("\nAccuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
    #         print("F1 (pos/neg) :", f_score)
    #         with open(log_file, "a") as file:
    #             file.write(f"mae: {mae}\n")
    #             file.write(f"corr: {corr}\n")
    #             file.write(f"mult_acc5: {mult_a5}")
    #             file.write(f"mult_acc7: {mult_a7}\n")
    #             file.write(f"\nClassification Report (pos/neg) : \n")
    #             file.write(classification_report(binary_truth, binary_preds, digits=5))
    #             file.write(f"\nAccuracy (pos/neg) {accuracy_score(binary_truth, binary_preds)}\n")
    #             file.write(f"F1 (pos/neg) : {f_score}\n")

    #     if mode=="train": # Jack Add 0531 pos/neg 也要提上日程
    #         pass
    #     if mode=="dev":
    #         print("Accuracy_val (pos/neg) ", accuracy_score(binary_truth, binary_preds))
    #         with open(log_file, "a") as file:
    #                 file.write(f"Accuracy_val (pos/neg) {accuracy_score(binary_truth, binary_preds)}\n")        
    #     if mode=="test":
    #         print("test_Accuracy (pos/neg) ", accuracy_score(binary_truth, binary_preds))
    #         with open(log_file, "a") as file:
    #                 file.write(f"Accuracy (pos/neg) {accuracy_score(binary_truth, binary_preds)}\n")

    #     # non-neg - neg
    #     binary_truth = (test_truth >= 0)
    #     binary_preds = (test_preds >= 0)

    #     f_score2 = f1_score((test_preds >= 0), (test_truth >= 0), average='weighted')

    #     if to_print:
    #         print("Classification Report (non-neg/neg) :\n")
    #         print(classification_report(binary_truth, binary_preds, digits=5))
    #         print("\nAccuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
    #         print("F1 (non-neg/neg) :", f_score2)
    #         with open(log_file, "a") as file:
    #             file.write(f"Classification Report (non-neg/neg): \n")
    #             file.write(classification_report(binary_truth, binary_preds, digits=5))
    #             file.write(f"\nAccuracy (non-neg/neg) {accuracy_score(binary_truth, binary_preds)}\n")
    #             file.write(f"F1 (non-neg/neg) : {f_score2}\n")
    #     if mode=="train":
    #         pass
    #     if mode=="dev":
    #         print("Accuracy_val (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
    #         with open(log_file, "a") as file:
    #                 file.write(f"Accuracy_val (non-neg/neg) {accuracy_score(binary_truth, binary_preds)}\n")        
    #     if mode=="test":
    #         print("F1 (non-neg/neg) :", f_score2)            
    #         print("test_Accuracy (non-neg/neg) ", accuracy_score(binary_truth, binary_preds))
    #         with open(log_file, "a") as file:
    #                 file.write(f"F1 (non-neg/neg) : {f_score2}\n")
    #                 file.write(f"test_Accuracy (non-neg/neg) {accuracy_score(binary_truth, binary_preds)}\n")

    #     return accuracy_score(binary_truth, binary_preds) # 试试 0524 accuracy = (pred.eq(y)).float().mean() # 返回的是 none-neg/neg

    # def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
    #         """
    #         Per-emotion multi-label metrics for the 6 Ekman emotions.
    #         y_true : raw intensities in [0,3], shape [N,6]
    #         y_pred : raw model logits,          shape [N,6]
    #         A true intensity > 0  -> ground-truth "present".
    #         A logit > 0 (== sigmoid > 0.5) -> predicted "present".
    #         """
    #         emotions = ["happy", "sad", "anger", "surprise", "disgust", "fear"]

    #         y_true = np.array(y_true).reshape(-1, len(emotions))
    #         y_pred = np.array(y_pred).reshape(-1, len(emotions))

    #         true_bin = (y_true > 0).astype(int)
    #         pred_bin = (y_pred > 0).astype(int)

    #         accs, f1s = [], []
    #         for i in range(len(emotions)):
    #             accs.append(accuracy_score(true_bin[:, i], pred_bin[:, i]))
    #             f1s.append(f1_score(true_bin[:, i], pred_bin[:, i],
    #                                 average="weighted", zero_division=0))
    #         mean_acc = float(np.mean(accs))
    #         mean_f1 = float(np.mean(f1s))

    #         if mode == "dev":
    #             print(f"[dev] mean emotion Acc: {mean_acc:.4f}  mean F1: {mean_f1:.4f}")
    #             with open(log_file, "a") as file:
    #                 file.write(f"[dev] mean emotion Acc: {mean_acc:.4f}  mean F1: {mean_f1:.4f}\n")

    #         if mode == "test" or to_print:
    #             header = f"{'emotion':10s} {'Acc':>8s} {'F1':>8s}"
    #             print("\n================ Per-emotion test results ================")
    #             print(header)
    #             for i, emo in enumerate(emotions):
    #                 print(f"{emo:10s} {accs[i]:8.4f} {f1s[i]:8.4f}")
    #             print(f"{'MEAN':10s} {mean_acc:8.4f} {mean_f1:8.4f}")
    #             with open(log_file, "a") as file:
    #                 file.write("\n================ Per-emotion test results ================\n")
    #                 file.write(header + "\n")
    #                 for i, emo in enumerate(emotions):
    #                     file.write(f"{emo:10s} {accs[i]:8.4f} {f1s[i]:8.4f}\n")
    #                 file.write(f"{'MEAN':10s} {mean_acc:8.4f} {mean_f1:8.4f}\n")

    #         return mean_acc

    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
        """
        Per-emotion DETECTION metrics for the 6 Ekman emotions.
        present  <=>  true intensity > 0  /  logit > 0 (== sigmoid > 0.5).
        Reports positive-class Precision / Recall / F1 (+ support, Acc for ref).
        Headline = MACRO-F1 (every emotion weighted equally).
        """
        from sklearn.metrics import (precision_score, recall_score,
                                     f1_score, accuracy_score)
        emotions = ["happy", "sad", "anger", "surprise", "disgust", "fear"]

        y_true = np.array(y_true).reshape(-1, len(emotions))
        y_pred = np.array(y_pred).reshape(-1, len(emotions))
        true_bin = (y_true > 0).astype(int)
        pred_bin = (y_pred > 0).astype(int)

        rows = []
        for i, emo in enumerate(emotions):
            t, p = true_bin[:, i], pred_bin[:, i]
            rows.append((
                emo,
                int(t.sum()),                                        # support
                accuracy_score(t, p),
                precision_score(t, p, pos_label=1, zero_division=0), # precision
                recall_score(t, p, pos_label=1, zero_division=0),    # recall
                f1_score(t, p, pos_label=1, zero_division=0),        # F1
            ))
        macro_f1 = float(np.mean([r[5] for r in rows]))
        macro_rec = float(np.mean([r[4] for r in rows]))

        def emit(write):
            write(f"{'emotion':10s} {'support':>8s} {'Acc':>7s} {'Prec':>7s} {'Recall':>7s} {'F1':>7s}")
            for emo, sup, acc, prec, rec, f1 in rows:
                write(f"{emo:10s} {sup:8d} {acc:7.3f} {prec:7.3f} {rec:7.3f} {f1:7.3f}")
            write(f"{'MACRO':10s} {'':>8s} {'':>7s} {'':>7s} {macro_rec:7.3f} {macro_f1:7.3f}")

        if mode == "dev":
            print(f"[dev] macro-F1: {macro_f1:.4f}  macro-Recall: {macro_rec:.4f}")
            with open(self.log_file, "a") as file:
                file.write(f"[dev] macro-F1: {macro_f1:.4f}  macro-Recall: {macro_rec:.4f}\n")

        if mode == "test" or to_print:
            title = "\n===== Per-emotion detection metrics (positive / 'present' class) ====="
            print(title)
            emit(print)
            with open(self.log_file, "a") as file:
                file.write(title + "\n")
                emit(lambda s: file.write(s + "\n"))

        return macro_f1


    def build(self, cuda=True):
        if self.model is None:
            self.model = getattr(new_models, self.train_config.model)(self.train_config)  # init the model

        # Final list
        for name, param in self.model.named_parameters():
            # Bert freezing customizations
            if self.train_config.data == "mosei":
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    # if layer_num <= (8):
                    #     param.requires_grad = False
                    if layer_num <= self.train_config.text_freeze_upto:
                        param.requires_grad = False
            if self.train_config.data == "ur_funny": # Jack Add
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    if layer_num <= (8):
                        param.requires_grad = False
            # elif self.train_config.data == "ur_funny": # Jack Change 0523 
            #     if "bert" in name:
            #         param.requires_grad = False
            if self.train_config.data == "mosi": # Jack Change 0523 
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    if layer_num <= (8):
                        param.requires_grad = False       
            if "bertmodel.embeddings" in name:
                param.requires_grad = False
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            # print('\t' + name, param.requires_grad)
        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)

    def model_input2output(self, batch):
        """
        get output from model input
        :param batch: batch
        :return: y_tilde: model predict output
                 y: true label
        """
        self.model.zero_grad()

        v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask,_segment_ids = batch # ([63, 64, 35]) ([63, 64, 74]) ([64, 1]) 64 ([64, 65]) 
        # ([42, 128, 47]) ([42, 128, 74]) ([128, 1]) 128 ([128, 44])
        v = to_gpu(v)
        a = to_gpu(a)
        y = to_gpu(y)

        bert_sent = to_gpu(bert_sent)
        bert_sent_type = to_gpu(bert_sent_type)
        bert_sent_mask = to_gpu(bert_sent_mask)

        y_tilde = self.model(v, a, l, bert_sent, bert_sent_type, bert_sent_mask)

        return y_tilde, y # [32,1] [32,1]