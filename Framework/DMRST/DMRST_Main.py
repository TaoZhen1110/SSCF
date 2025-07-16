import torch
from transformers import AutoTokenizer, AutoModel
from DMRST.DMRST_ParsingNet import ParsingNet  # 你自己的 ParsingNet 模型实现
import networkx as nx
# from networkx.drawing.nx_agraph import graphviz_layout
from networkx.drawing.nx_pydot import graphviz_layout
import re
import matplotlib.pyplot as plt
import os



####################################  推理修辞关系  ####################################
def inference_single(model, tokenizer, sentence):
    """
        对单个句子进行推理，生成 EDU 分段和修辞树。
        Args:
            model: ParsingNet 模型
            tokenizer: BERT 分词器
            sentence: 输入的句子 (str)
        Returns:
            tokenized_sentence: 分词后的句子
            segmentation_pred: EDU 分段结果
            tree_parsing_pred: 修辞树生成结果
        """
    with torch.no_grad():
        # 分词
        tokenized_sentence = tokenizer.tokenize(sentence, add_special_tokens=False)
    

    # 模型推理
    _, _, SPAN_batch, _, predict_EDU_breaks = model.TestingLoss(
        [tokenized_sentence],
        input_EDU_breaks=None,
        LabelIndex=None,
        ParsingIndex=None,
        GenerateTree=True,
        use_pred_segmentation=True
    )


    # 提取修辞树（SPAN_batch 通常是修辞树的文本格式表示）
    rst_tree = SPAN_batch[0][0]

    # 提取 EDU 分段
    edus = []
    prev_break = 0
    for edu_break in predict_EDU_breaks[0]:
        edus.append((prev_break, edu_break))
        prev_break = edu_break + 1

    # 解析修辞树，生成节点和边
    nodes = []
    edges = []
    node_id_counter = 1

    # 使用正则解析修辞树的节点和关系
    relations = rst_tree.strip("[]").split(") (")
    relations = [rel.strip("()") for rel in relations]

    node_mapping = {}  # 存储 Parent 和 Child 节点的映射
    
    for relation in relations:
        match = re.match(r'(\d+):(\w+)=(\w+):(\d+),(\d+):(\w+)=(\w+):(\d+)', relation)
        if match:
            parent_id = int(match.group(1))
            parent_type = match.group(2)
            relation_type = match.group(3)
            child1_id = int(match.group(4))
            child2_id = int(match.group(5))
            child1_type = match.group(6)
            child2_type = match.group(7)

            # 添加节点
            if parent_id not in node_mapping:
                nodes.append({
                    "id": node_id_counter,
                    "type": parent_type,
                    "text": " ".join(tokenized_sentence[edus[parent_id - 1][0]:edus[parent_id - 1][1] + 1])
                })
                node_mapping[parent_id] = node_id_counter
                node_id_counter += 1

            if child1_id not in node_mapping:
                nodes.append({
                    "id": node_id_counter,
                    "type": child1_type,
                    "text": " ".join(tokenized_sentence[edus[child1_id - 1][0]:edus[child1_id - 1][1] + 1])
                })
                node_mapping[child1_id] = node_id_counter
                node_id_counter += 1

            if child2_id not in node_mapping:
                nodes.append({
                    "id": node_id_counter,
                    "type": child2_type,
                    "text": " ".join(tokenized_sentence[edus[child2_id - 1][0]:edus[child2_id - 1][1] + 1])
                })
                node_mapping[child2_id] = node_id_counter
                node_id_counter += 1

            # 添加边
            edges.append({
                "from": node_mapping[parent_id],
                "to": node_mapping[child1_id],
                "relation": relation_type
            })
            edges.append({
                "from": node_mapping[parent_id],
                "to": node_mapping[child2_id],
                "relation": relation_type
            })

        # 返回结构化结果
    output_dict = {
        # "tokens": tokenized_sentence,
        # "edus": edus,
        "nodes": nodes,
        "edges": edges
    }
    return output_dict





def DMRST_main(text):

    # 模型路径和参数
    MODEL_PATH = "/mnt/public/code/taozhen/LLM_Review_Detection/Framework/DMRST/multi_all_checkpoint.torchsave"  # 预训练模型路径
    BERT_MODEL_NAME = "/mnt/public/code/taozhen/LLM_Review_Detection/Framework/DMRST/xlm-roberta-base"  # 预训练的 BERT 模型
    BATCH_SIZE = 1  # 每次处理的句子数

    # 加载 BERT 模型和分词器
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, use_fast=True)
    bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).cuda()

    # 冻结 BERT 参数
    for param in bert_model.parameters():
        param.requires_grad = False

    # 加载 ParsingNet 模型
    model = ParsingNet(bert_model, bert_tokenizer=bert_tokenizer).cuda()
    model.load_state_dict(torch.load(MODEL_PATH), strict=False)  # 加载已保存的权重
    model.eval()  # 设置为评估模式



    result = inference_single(model, bert_tokenizer, text)

    return result


# text = """
# Artificial Intelligence (AI) has evolved at an unprecedented pace over the past few decades, transforming from a concept in science fiction into a fundamental part of modern society. From simple rule-based systems to highly sophisticated deep learning models, AI has changed how we interact with technology, solve complex problems, and even perceive intelligence itself.
# """
# print(DMRST_main(text))