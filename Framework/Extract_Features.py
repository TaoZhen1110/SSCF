from multiprocessing import Pool
import multiprocessing as mp
import nltk
import numpy as np
from nltk import pos_tag, word_tokenize
from collections import Counter
from scipy.stats import entropy
import string
import textstat
import re
import spacy
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import os
from nltk.sentiment import SentimentIntensityAnalyzer
from DMRST.DMRST_ParsingNet import ParsingNet 
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Data
import json
from tqdm import tqdm


#################################################  Grammatical Features  #################################################

########################################### 一、 Statistic Features

def Extract_Statistic_Features(Review_Text):

    ########### 1. 基本统计
    # 句子分割
    sentences = nltk.sent_tokenize(Review_Text)
    # 词汇分割
    words = nltk.word_tokenize(Review_Text)
    words = [word for word in words if word.isalpha()]   # 过滤掉标点符号
    # 1.1 句子数量
    sentences_number= len(sentences)
    # 1.2 词语数量
    words_number = len(words)
    # 1.3 平均词长（排除空格和标点符号，计算字母和数字）
    character_count_alnum_number = sum(1 for char in Review_Text if char.isalnum())
    average_word_length = character_count_alnum_number / words_number if words_number > 0 else 0
    # 1.4 平均句长（词数/句数）
    average_sentence_length = words_number / sentences_number if sentences_number > 0 else 0
    # 1.5 句子长度的标准差
    sentence_lengths = [len(nltk.word_tokenize(sentence)) for sentence in sentences]
    sentence_lengths = [length for length, sentence in zip(sentence_lengths, sentences) if
                        any(char.isalpha() for char in sentence)]
    sentence_length_std = np.std(sentence_lengths) if sentence_lengths else 0
    sentence_length_std = round(sentence_length_std, 4)
    Basic_Statistic_features = [sentences_number, words_number, average_word_length, average_sentence_length, sentence_length_std]

    ########### 2. 词性（Part-of-Speech, POS）分布
    # Tokenization 和 POS Tagging
    tokens = word_tokenize(Review_Text)
    pos_tags = pos_tag(tokens)
    # 过滤掉标点符号（POS 标签为标点符号的特定类型）
    filtered_pos_tags = [(word, tag) for word, tag in pos_tags if tag.isalpha()]
    # 统计词性分布
    pos_counts = Counter(tag for word, tag in filtered_pos_tags)
    # 计算总词数
    total_tokens = sum(pos_counts.values())
    # 计算词性占比
    pos_distribution = {tag: count / total_tokens for tag, count in pos_counts.items()}
    # 转化为固定顺序的特征向量
    # 定义所有可能的词性标签（可根据任务需求调整，常用词性包括 'NN', 'VB', 'JJ' 等）
    all_pos_tags = ['NN', 'VB', 'JJ', 'RB', 'DT', 'IN', 'PRP']  # 示例词性列表
    # 2.1 构建特征向量
    pos_distribution_vector = [pos_distribution.get(tag, 0) for tag in all_pos_tags]
    # 2.2 计算熵（Entropy）
    pos_entropy = [entropy(list(pos_distribution.values()))]
    Pos_features = pos_distribution_vector + pos_entropy

    ########### 3. 词汇多样性特征
    tokens_without_punctuation = [token for token in tokens if token.isalpha()]
    total_tokens = len(tokens_without_punctuation)
    unique_tokens = set(tokens_without_punctuation)
    type_count = len(unique_tokens)
    # 3.1 Root TTR
    rttr = type_count / np.sqrt(total_tokens)
    # 3.2 Hapax Legomena Ratio（仅出现一次的词占比）
    hapax_legomena = (
        len([word for word, count in Counter(tokens).items() if count == 1]) / total_tokens
        if total_tokens > 0
        else 0
    )
    # 3.3 Shannon Entropy (词汇分布熵)
    token_counts = Counter(tokens)
    if total_tokens > 0:
        proportions = np.array(list(token_counts.values())) / total_tokens
        shannon_entropy = -np.sum(proportions * np.log2(proportions))
    else:
        shannon_entropy = 0
    Vocabulary_Diversity_features = [rttr, hapax_legomena, shannon_entropy]

    ########### 4. 标点符号使用
    # 提取标点符号
    punctuations = [char for char in Review_Text if char in string.punctuation]
    # 统计标点符号总数
    total_punctuations = len(punctuations)
    # 4.1 统计特殊标点的比例（问号、感叹号、冒号）
    special_punctuation = ["?", "!", ":"]
    special_counts = {p: punctuations.count(p) for p in special_punctuation}
    Special_Proportions_features = [count / total_punctuations for p, count in special_counts.items()]

    ########### 5. 文本可读性
    flesch_reading_ease = textstat.flesch_reading_ease(Review_Text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(Review_Text)
    gunning_fog = textstat.gunning_fog(Review_Text)
    dale_chall_score = textstat.dale_chall_readability_score(Review_Text)
    automated_readability = textstat.automated_readability_index(Review_Text)
    text_standard = textstat.text_standard(Review_Text)
    grades = [int(grade) for grade in re.findall(r'\d+', text_standard)]
    if grades:
        average_grade = sum(grades) / len(grades)
    else:
        average_grade = 0  # 如果没有数字，可能是未分类文本
    Readability_features = [flesch_reading_ease, flesch_kincaid_grade, gunning_fog, dale_chall_score, automated_readability, average_grade]

    Statistic_Features = Basic_Statistic_features + Pos_features + Vocabulary_Diversity_features + Special_Proportions_features + Readability_features

    return Statistic_Features


########################################### 二、 Entity Features
def Extract_Entity_Features(Paper_Text, Review_Text, nlp):
    # 定义重要实体列表
    IMPORTANT_ENTITIES = {"PRODUCT", "EVENT", "ORG", "WORK_OF_ART", "CARDINAL"}
    # 提取论文及评论的实体和类别分布
    Paper_Doc = nlp(Paper_Text)
    Review_Doc = nlp(Review_Text)
    # 提取论文及评论的所有实体
    Paper_entities = set([ent.text for ent in Paper_Doc.ents])
    Paper_entity_labels = [ent.label_ for ent in Paper_Doc.ents]
    Review_entities = set([ent.text for ent in Review_Doc.ents])
    Review_entity_labels = [ent.label_ for ent in Review_Doc.ents]
    # 统计类别分布
    Paper_label_counts = Counter(Paper_entity_labels)
    Paper_total_labels = sum(Paper_label_counts.values())
    Paper_distribution = {label: count / Paper_total_labels for label, count in Paper_label_counts.items()}
    Review_label_counts = Counter(Review_entity_labels)
    Review_total_labels = sum(Review_label_counts.values())
    Review_distribution = {label: count / Review_total_labels for label, count in Review_label_counts.items()}
    # 1. 计算实体覆盖率
    common_entities = Paper_entities.intersection(Review_entities)
    coverage_rate = round(len(common_entities) / len(Paper_entities) if Paper_entities else 0, 4)
    # 2. 计算重要实体覆盖率
    important_paper_entities = {ent for ent in Paper_entities if ent in IMPORTANT_ENTITIES}
    important_common_entities = important_paper_entities.intersection(Review_entities)
    important_coverage_rate = round(len(important_common_entities) / len(important_paper_entities) if important_paper_entities else 0, 4)
    # 3. 计算类别分布相似度
    all_labels = set(Paper_distribution.keys()).union(set(Review_distribution.keys()))
    vec_paper = [Paper_distribution.get(label, 0) for label in all_labels]
    vec_review = [Review_distribution.get(label, 0) for label in all_labels]
    if sum(vec_paper) == 0 or sum(vec_review) == 0:
        similarity = 0  # You can adjust this to 1 if you prefer treating them as identical
    else:
        similarity = 1 - cosine(vec_paper, vec_review)

    Entity_Features = [coverage_rate, important_coverage_rate, similarity]

    return Entity_Features




#################################################  Semantic Features  #################################################

########################################### 三、 Perplexity Features

########################################### 四、 Sentiment Features

# 提取情绪主函数
def Extract_Sentiment_Features(sia, Review_Text):

    # 1. 提取情绪向量
    scores = sia.polarity_scores(Review_Text)
    sentiment_vector = [scores['pos'], scores['neg'], scores['neu'], scores['compound']]

    # 2. 构建情绪特征
    positive, negative, neutral, _ = sentiment_vector
    balance = abs(positive - negative) / (positive + negative + neutral + 1)
    senti_feature = [positive, negative, neutral, balance]       
    input_features = np.array(senti_feature)        # 转换为数组
    senti_tensor = torch.tensor(input_features, dtype=torch.float).unsqueeze(0)

    return senti_tensor


########################################### 五、 Structure Features
### 推理修辞关系
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


### DMRST主函数
def DMRST_main(text, DMRST_MODEL_PATH, DMRST_tokenizer, DMRST_model):

    # 冻结 BERT 参数
    for param in DMRST_model.parameters():
        param.requires_grad = False

    # 加载 ParsingNet 模型
    model = ParsingNet(DMRST_model, bert_tokenizer=DMRST_tokenizer).cuda()
    model.load_state_dict(torch.load(DMRST_MODEL_PATH), strict=False)  # 加载已保存的权重
    model.eval()  # 设置为评估模式

    result = inference_single(model, DMRST_tokenizer, text)

    return result


### 构建边特征 (用Roberta对"边"编码)
def encode_relation_with_roberta(relation, tokenizer, model):

    inputs = tokenizer(relation, return_tensors="pt", padding=True, truncation=True).to('cuda')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state[:, 0, :].cpu().squeeze(0)  # 取 [CLS] 向量作为特征


### GNN-Transformer提取图特征
class GraphTransformerModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, heads=4):
        super(GraphTransformerModel, self).__init__()
        # 第一个 TransformerConv 层，将输入的节点特征转换为 hidden_dim 维度
        self.conv1 = TransformerConv(in_dim, hidden_dim, heads=heads, edge_dim=num_relations)
        # 第二个 TransformerConv 层，将第一个 TransformerConv 层的输出进行处理
        self.conv2 = TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, edge_dim=num_relations)
        # 全连接层，将特征映射到最终的输出维度
        self.fc_out = torch.nn.Linear(hidden_dim * heads, out_dim)

    def forward(self, x, edge_index, edge_attr, batch=None):
        # 第一次卷积操作
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)  # 激活函数
        # 第二次卷积操作
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)  # 激活函数
        # 通过全连接层进行最后的映射
        x = self.fc_out(x)
        # 添加全局池化
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)  # 假设单图，batch 全为 0
        x = global_mean_pool(x, batch)  # 输出图级特征

        return x

    
def Extract_Stucture_Features(Review_Text, Roberta_tokenizer, Roberta_model, 
                              DMRST_MODEL_PATH, DMRST_tokenizer, DMRST_model, Graph_model):

    # 检查输入是否为空或只包含特殊字符
    if not Review_Text or not any(c.isalnum() for c in Review_Text):
        return torch.zeros(1, 768)  # 返回维度为 [1, 768] 的全零张量
    
    # 1. 获取修辞树数据
    data = DMRST_main(Review_Text, DMRST_MODEL_PATH, DMRST_tokenizer, DMRST_model)
    
    if not data['nodes'] or not data['edges']:
        inputs = Roberta_tokenizer(Review_Text, return_tensors="pt", max_length=64, truncation=True, padding=True).to('cuda')
        with torch.no_grad():
            outputs = Roberta_model(**inputs)
            node_features = outputs.last_hidden_state[:, 0, :].cpu()  # 取 [CLS] token 的特征向量

        
        return node_features
    
    else:
        # 2. 提取节点文本特征
        nodes = data['nodes']
        texts = [node['text'].replace(' ', '').replace('▁', ' ') for node in nodes]
        inputs = Roberta_tokenizer(texts, return_tensors="pt", max_length=64, truncation=True, padding=True).to('cuda')
        with torch.no_grad():
            outputs = Roberta_model(**inputs)
            node_features = outputs.last_hidden_state[:, 0, :].cpu()
        
        # 3. 处理边索引
        edges = data['edges']
        edge_index = torch.tensor([[edge['from'] - 1, edge['to'] - 1] for edge in edges], dtype=torch.long).t()

        # 4. 构建边特征 (用Roberta对 边 编码)
        relation_types = list(set(edge['relation'] for edge in edges))
        relation_embeddings = {rel: encode_relation_with_roberta(rel, Roberta_tokenizer, Roberta_model) for rel in relation_types}
        edge_attr = torch.stack([relation_embeddings[edge['relation']] for edge in edges])

        # 5. 处理节点类型
        predefined_types = ['Elaboration', 'Temporal', 'Satellite', 'Joint', 'Evaluation', 
                        'Explanation', 'span', 'Contrast', 'Nucleus', 'Cause', 
                        'Enablement', 'Attribution']
        type_to_idx = {node_type: idx for idx, node_type in enumerate(predefined_types)}
        type_to_idx["other"] = len(predefined_types)
        num_classes = len(predefined_types) + 1  # 总类别数，包括 "other"
        one_hot_lookup = torch.eye(num_classes)  # 创建一个大小为 (num_classes, num_classes) 的单位矩阵

        node_type_indices = torch.tensor(
        [type_to_idx.get(node['type'], type_to_idx["other"]) for node in data['nodes']], dtype=torch.long
        )     # 获取节点类型的索引

        node_type_features = one_hot_lookup[node_type_indices]     # 对每个节点选择对应的 One-Hot 编码

        # 6. 最后提取图特征
        Node_All_Features = torch.cat([node_features, node_type_features], dim=1)     # 节点特征汇总
        graph_data = Data(x=Node_All_Features, edge_index=edge_index, edge_attr=edge_attr)  # 创建Data对象
        with torch.no_grad():
            stucture_features = Graph_model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)

        return stucture_features




#################################################  All Features  #################################################
def Extract_All_Features(Review_Text, Paper_Text, nlp, sia, Split_Review_Text, Roberta_Tokenizer, Roberta_Model, 
                         DMRST_MODEL_PATH, DMRST_Tokenizer, DMRST_Model, Graph_model):

    ##### 语法特征提取
    Statistic_Features = Extract_Statistic_Features(Review_Text)     # 一、 Statistic Features
    Entity_Features = Extract_Entity_Features(Paper_Text, Review_Text, nlp)   # 二、 Entity Features

    ##### 语义特征提取
    Semantic_Features = []
    for section, text in Split_Review_Text.items():
        sentiment_feature = Extract_Sentiment_Features(sia, text)
        stucture_feature = Extract_Stucture_Features(text, Roberta_Tokenizer, Roberta_Model, 
                                                    DMRST_MODEL_PATH, DMRST_Tokenizer, DMRST_Model, Graph_model)
        semantic_feature = torch.cat([sentiment_feature, stucture_feature], dim=1)
        # print(semantic_feature.shape)
        # 将每个拼接的 semantic_feature 添加到列表中
        Semantic_Features.append(semantic_feature)


    Semantic_vector = torch.stack(Semantic_Features, dim=1).squeeze(0)
    Semantic_list = Semantic_vector.tolist()

    ##### 汇总特征
    All_Features = {
        "Statistic_Features": Statistic_Features,
        "Entity_Features": Entity_Features,
        "Semantic_list": Semantic_list
    }

    return All_Features



def process_data(data):

    torch.manual_seed(1234)  # 固定种子，设置随机种子
    # 加载 spaCy 的预训练模型
    nlp = spacy.load("en_core_web_sm")
    # 初始化 VADER，nltk
    sia = SentimentIntensityAnalyzer()
    
    # Roberta模型设置
    model_name = "/mnt/public/code/taozhen/LLM_Review_Detection/Framework/DMRST/roberta-base/"
    Roberta_Tokenizer = AutoTokenizer.from_pretrained(model_name)
    Roberta_Model = AutoModel.from_pretrained(model_name).cuda()

    # DMRST模型设置
    DMRST_MODEL_PATH = "/mnt/public/code/taozhen/LLM_Review_Detection/Framework/DMRST/multi_all_checkpoint.torchsave"  # 预训练模型路径
    DMRST_MODEL_NAME = "/mnt/public/code/taozhen/LLM_Review_Detection/Framework/DMRST/xlm-roberta-base"  # 预训练的 BERT 模型
    DMRST_Tokenizer = AutoTokenizer.from_pretrained(DMRST_MODEL_NAME, use_fast=True)
    DMRST_Model = AutoModel.from_pretrained(DMRST_MODEL_NAME).cuda()

    # 图神经网络设置
    in_dim = 768 + 13  # 节点特征维度: 768 (RoBERTa) + 13 (One-Hot)
    hidden_dim = 512   # 隐藏层维度
    out_dim = 768       # 输出特征维度
    num_relations = 768  # 边特征维度 (RoBERTa 的 [CLS] 向量)
    Graph_model = GraphTransformerModel(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim, num_relations=num_relations) # 实例化模型


    new_item = {}
    for key,value in data.items():
        if "Review" not in key and "Paper_Text" not in key:
            new_item[key] = value

        if "Review" in key and "Split" not in key:
            Review_Text = data[key]
            Paper_Text = data["Paper_Text"]
            Review_Split = data[key+"_Split"]
            All_Features = Extract_All_Features(Review_Text=Review_Text, Paper_Text=Paper_Text, nlp=nlp, sia=sia, 
                                 Split_Review_Text=Review_Split, Roberta_Tokenizer=Roberta_Tokenizer, Roberta_Model=Roberta_Model, 
                                 DMRST_MODEL_PATH=DMRST_MODEL_PATH, DMRST_Tokenizer=DMRST_Tokenizer, DMRST_Model=DMRST_Model, 
                                 Graph_model=Graph_model)
            new_item[key] = All_Features
    
    return new_item



def process_line_with_retry(line, max_attempts=3):

    for attempt in range(1, max_attempts + 1):
        try:
            return process_data(line)
        except Exception as e:
            print(f"处理失败，尝试次数 {attempt}/{max_attempts}: {e}")
            if attempt == max_attempts:
                # 达到最大尝试次数，可以选择返回一个特定的错误标记，或者抛出异常
                return None  # 或者 raise


def save_data(data, Split_File):
    """
    将处理后的数据立即保存到文件中。
    """
    with open(Split_File, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")



#########################################################################################################################
if __name__ == "__main__":
    mp.set_start_method('spawn')

    # ###################################### 重要参数设置 ######################################
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.manual_seed(1234)  # 固定种子，设置随机种子


    ###################################### 提取特征代码 ######################################

    processes = 3
    p = Pool(processes=processes)

    name_list = ["GLM", "Internlm"]   # "GLM"

    for name in name_list:
        Input_Original_File = f"/mnt/public/code/taozhen/LLM_Review_Detection/Dataset/Final_Dataset/{name}/Data2_1.json"
        Output_Split_File = f"/mnt/public/code/taozhen/LLM_Review_Detection/Dataset/Final_Dataset/{name}/Data2_2.json"

        # 如果输出文件不存在，创建并处理数据
        if not os.path.exists(Output_Split_File):
            print(f"Creating {Output_Split_File}")

            # 读取输入数据
            with open(Input_Original_File, 'r', encoding='utf-8') as file:
                data_list = [json.loads(line.strip()) for line in file]

            # 使用 imap_unordered 获取迭代器，允许在任务完成时立即处理结果
            with tqdm(total=len(data_list), desc="Processing Data") as progress_bar:
                for result in p.imap_unordered(process_line_with_retry, data_list):
                    if result is not None:
                        save_data(result, Output_Split_File)
                        progress_bar.update(1)  # 更新进度条
        
        else:
            print(f"Loading {Output_Split_File} to check missing items")
            existing_ids = set()
            missing_items = []

            # 读取已有的文件，并记录已有的 ID
            with open(Output_Split_File, 'r', encoding='utf-8') as existing_file:
                for line in existing_file:
                    data = json.loads(line)
                    existing_ids.add(data['Paper_ID'])
            
            # 读取源数据文件，查找缺失的条目
            with open(Input_Original_File, 'r', encoding='utf-8') as original_file:
                data_list = [json.loads(line.strip()) for line in original_file]
            
            # 识别缺失项
            for item in data_list:
                if item['Paper_ID'] not in existing_ids:
                    missing_items.append(item)
            
            # 处理缺失的条目
            with tqdm(total=len(missing_items), desc="Processing Missing Items") as progress_bar:
                for result in p.imap_unordered(process_line_with_retry, missing_items):
                    if result is not None:
                        save_data(result, Output_Split_File)
                        progress_bar.update(1)
            
    
    p.close()
    p.join()
