import argparse
import os
import json
from Dataloader import Data_Loader, dataset_loader
from model import Review_Detection_Framework
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score, matthews_corrcoef, balanced_accuracy_score



def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=str, default='0,1')
    parser.add_argument('-test_batch_size', type=int, default=64)
    parser.add_argument('-test_dataset_path', type=str,
                        default="/mnt/public/code/taozhen/LLM_Review_Detection/Dataset/Model_Dataset/Feature_Dataset/test1.json")
    parser.add_argument("-model_path", type=str,
                        default='/mnt/public/code/taozhen/LLM_Review_Detection/Experiment4/Model2/run_3/model.pth')
    
    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    ###################  Dataset prepare #################
    test_data = []
    with open(args.test_dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            test_data.append(data)
            
    test_loader = dataset_loader(dataset=Data_Loader(jsondata=test_data), batch_size=args.test_batch_size, shuffle=True)

    ########################  Model ###########################
    model = Review_Detection_Framework()
    model = nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(args.model_path))

    ######################## test model ###########################
    all_preds = []  # 存储所有预测的标签
    all_labels = []  # 存储所有真实标签

    model.eval()
    for step, sample_batched in enumerate(test_loader):
        sample_batched = {key: value.cuda() for key, value in sample_batched.items()}
        Grammar_Feature = sample_batched["Grammar_Feature"]
        Semantic_Feature = sample_batched["Semantic_Feature"]
        Label = sample_batched["Label"]

        with torch.no_grad():
            Output = model(Grammar_Feature, Semantic_Feature)

        # 使用 argmax 来选择类别
        predicted = torch.argmax(Output, dim=1)  # 获取每个样本预测的类别

        all_preds.extend(predicted.cpu().numpy())  # 保存预测结果
        all_labels.extend(Label.cpu().numpy())  # 保存真实标签


    precision_per_class = precision_score(all_labels, all_preds, average=None)  # 针对每个类别
    recall_per_class = recall_score(all_labels, all_preds, average=None)  # 针对每个类别
    f1_per_class = f1_score(all_labels, all_preds, average=None)  # 针对每个类别

    # 计算不区分类别的 F1-score（macro 或 weighted）
    f1_macro = f1_score(all_labels, all_preds, average='macro')  # 不区分类别的 F1-score（宏平均）
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')  # 不区分类别的 F1-score（加权平均）




    # precision = precision_score(all_labels, all_preds, average='binary')  # 'binary' 用于二分类任务
    # recall = recall_score(all_labels, all_preds, average='binary')  # 'binary' 用于二分类任务
    f1 = f1_score(all_labels, all_preds, average='binary')
    roc_auc = roc_auc_score(all_labels, all_preds)
    pr_auc = average_precision_score(all_labels, all_preds)
    mcc = matthews_corrcoef(all_labels, all_preds)
    # balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)




    # print(f'Accuracy: {accuracy:.5f}')
    # print(f'Precision: {precision:.5f}')
    # print(f'Recall: {recall:.5f}')


    print(f'Precision (Class 0): {precision_per_class[0]:.5f}, Precision (Class 1): {precision_per_class[1]:.5f}')
    print(f'Recall (Class 0): {recall_per_class[0]:.5f}, Recall (Class 1): {recall_per_class[1]:.5f}')
    print(f'F1-Score (Class 0): {f1_per_class[0]:.5f}, F1-Score (Class 1): {f1_per_class[1]:.5f}')


    # 输出不区分类别的 F1-score
    print(f'F1-Score (Macro Average): {f1_macro:.5f}')
    print(f'F1-Score (Weighted Average): {f1_weighted:.5f}')
    
    print(f'ROC-AUC: {roc_auc:.5f}')
    print(f'PR-AUC: {pr_auc:.5f}')
    print(f'MCC: {mcc:.5f}')
    # print(f'Balanced Accuracy: {balanced_accuracy:.5f}')


if __name__ == '__main__':
    args = get_arguments()
    main(args)