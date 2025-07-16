import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import glob
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter
import socket
import json
from Dataloader import Data_Loader, dataset_loader
from model import Review_Detection_Framework
import time
from sklearn.metrics import accuracy_score, f1_score


##### 设置主要参数
def get_argument():
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=str, default='0,1')
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('-resume_epoch', type=int, default=0)

    parser.add_argument('-train_dataset_path', type=str,
                        default="/mnt/public/code/taozhen/LLM_Review_Detection/Dataset/Model_Dataset/Feature_Dataset/train1.json")
    parser.add_argument('-val_dataset_path', type=str,
                        default="/mnt/public/code/taozhen/LLM_Review_Detection/Dataset/Model_Dataset/Feature_Dataset/val1.json")
    parser.add_argument('-train_batch_size', type=int, default=256)
    parser.add_argument('-val_batch_size', type=int, default=64)

    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-weight_decay', type=float, default=1e-4)
    parser.add_argument('-log_every', type=int, default=100)
    parser.add_argument('-naver_grad', type=int, default=1)
    

    return parser.parse_args()


##### 初始化神经网络中线性层的权重和偏置
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


##### 在训练神经网络时设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

##### 定义Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 使用Softmax将logits转化为概率
        inputs = F.softmax(inputs, dim=1)
        
        # 确保targets是Long类型
        targets = targets.long()
        
        # 获取每个样本的预测概率
        p_t = inputs.gather(dim=1, index=targets.unsqueeze(1))
        
        # 计算Focal Loss
        loss = -self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

##### 主函数
def main(args):

    #################  Setting Gpu and seed  #################
    setup_seed(1234)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    #################  Train model save setting  #################
    save_dir_root = os.path.dirname(os.path.abspath(__file__))

    # os.path.abspath:取当前文件的绝对路径（完整路径）
    # os.path.dirname:去掉文件名，返回目录

    if args.resume_epoch != 0:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    
    save_dir = os.path.join(save_dir_root, 'run_' + str(run_id))
    log_dir = os.path.join(save_dir, datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    ###################  Dataset prepare #################
    train_data = []
    with open(args.train_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            train_data.append(data)
            
    train_loader = dataset_loader(dataset=Data_Loader(jsondata=train_data), batch_size=args.train_batch_size, shuffle=True)
    
    val_data = []
    with open(args.val_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            val_data.append(data)

    val_loader = dataset_loader(dataset=Data_Loader(jsondata=val_data), batch_size=args.val_batch_size, shuffle=False)

    num_iter_tr = len(train_loader)
    nitrs = args.resume_epoch * num_iter_tr
    nsamples = 0


    ########################  Model ###########################
    model = Review_Detection_Framework()
    model.apply(init_weights)
    model = nn.DataParallel(model).cuda()

    ##################  优化 Setting #################
    parameters_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters_update, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # #################  Loss function  #################
    # loss_function = nn.CrossEntropyLoss()
    loss_function = FocalLoss(alpha=1, gamma=2, reduction='mean')

    # #################  Train Model  ##################
    recent_losses = []
    aveGrad = 0
    best_f1 = 0.0
    start_time = time.time()

    for epoch in tqdm(range(args.resume_epoch, args.epochs)):
        model.train()
        count1 = 0
        epoch_train_losses = []

        for step, sample_batched in enumerate(train_loader):
            sample_batched = {key: value.cuda() for key, value in sample_batched.items()}
            Grammar_Feature = sample_batched["Grammar_Feature"]
            Semantic_Feature = sample_batched["Semantic_Feature"]
            Label = sample_batched["Label"]

            Output = model(Grammar_Feature, Semantic_Feature)
            Loss = loss_function(Output, Label)

            count1 += Label.size(0)
            trainloss = Loss.item()
            epoch_train_losses.append(trainloss)

            if len(recent_losses) < args.log_every:
                recent_losses.append(trainloss)
            else:
                recent_losses[nitrs % len(recent_losses)] = trainloss
            # Backward the averaged gradient
            Loss.backward()
            aveGrad += 1
            nitrs += 1
            nsamples += args.train_batch_size

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % args.naver_grad == 0:  # args.naver_grad =1
                optimizer.step()  # 这个方法会更新所有的参数
                optimizer.zero_grad()
                aveGrad = 0

            if nitrs % args.log_every == 0:  # log_every=40
                meanloss1 = sum(recent_losses) / len(recent_losses)
                print('epoch: %d step: %d count: %d trainloss: %.5f timecost:%.2f secs' %
                      (epoch, step, count1, meanloss1, time.time() - start_time))
                writer.add_scalar('data/trainloss', meanloss1, nsamples)

        meanloss2 = sum(epoch_train_losses) / len(epoch_train_losses)
        print('epoch: %d meanloss: %.5f' % (epoch, meanloss2))
        writer.add_scalar('data/epochloss', meanloss2, nsamples)

        scheduler.step()


        ######################## eval model ###########################
        print("######## val data ########")
        epoch_val_losses = []
        count2 = 0
        all_preds = []  # 存储所有预测的标签
        all_labels = []  # 存储所有真实标签

        model.eval()

        for step, sample_batched in enumerate(val_loader):
            sample_batched = {key: value.cuda() for key, value in sample_batched.items()}
            Grammar_Feature = sample_batched["Grammar_Feature"]
            Semantic_Feature = sample_batched["Semantic_Feature"]
            Label = sample_batched["Label"]

            with torch.no_grad():
                Output = model(Grammar_Feature, Semantic_Feature)
            
            Loss = loss_function(Output, Label)
            
            valloss = Loss.item()
            epoch_val_losses.append(valloss)

            count2 += Label.size(0)

            # 使用 argmax 来选择类别
            predicted = torch.argmax(Output, dim=1)  # 获取每个样本预测的类别


            # 保存所有标签和预测的值，用于计算 F1 score
            all_preds.extend(predicted.cpu().numpy())  # 保存预测结果
            all_labels.extend(Label.cpu().numpy())  # 保存真实标签


        ########## 计算损失函数 ##########
        mean_valLoss = sum(epoch_val_losses) / len(epoch_val_losses)

        # 计算准确率
        Accuracy = accuracy_score(all_labels, all_preds)

        # 计算 F1 score
        F1 = f1_score(all_labels, all_preds, average='binary')  # binary 是二分类任务的选项


        print('Validation:')
        print('epoch: %d, numtext: %d valLoss: %.5f Accuracy: %.5f F1 score: %.5f' % (
            epoch, count2, mean_valLoss, Accuracy, F1))

        # 记录在 TensorBoard
        writer.add_scalar('data/valloss', mean_valLoss, count2)
        writer.add_scalar('data/accuracy', Accuracy, count2)  # 添加准确率到 TensorBoard
        writer.add_scalar('data/F1_score', F1, count2)  # 添加 F1 score 到 TensorBoard


        ################   Save Pth  ################

        # 如果当前的 accuracy 和 F1 score 都超过之前的最佳值，则保存模型
        if F1 > best_f1:  # 同时检查准确率和 F1 是否超过最佳值
            save_path = os.path.join(save_dir, 'model' + '.pth')
            torch.save(model.state_dict(), save_path)
            print("Save model at {}\n".format(save_path))
            best_f1 = F1


if __name__ == "__main__":
    args = get_argument()
    main(args)