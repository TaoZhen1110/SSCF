import torch
import torch.nn as nn


####################################### 双向 LSTM四个部分的语义特征
class Optimized_BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Optimized_BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 双向

    def forward(self, x):
        # 获取 LSTM 输出
        lstm_out, _ = self.bilstm(x)
        # 对所有时间步的输出做平均池化，得到一个固定长度的向量
        pooled_out = lstm_out.mean(dim=1)  # dim=1 表示按时间维度（seq_len）计算平均值
        # 通过全连接层进行分类
        updated_features = self.fc(pooled_out)
    
        return updated_features



class Self_Attention_Module(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.5):
        super(Self_Attention_Module, self).__init__()
        self.multihead_attenion = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.multihead_attenion(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x


class Review_Detection_Framework(nn.Module):
    def __init__(self):
        super(Review_Detection_Framework, self).__init__()
        self.BiLSTM = Optimized_BiLSTMModel(772, 256, 256, 2)
        self.SelfAttention = Self_Attention_Module(256, 8, 128)

        # 统一维度
        self.grammar_projection = nn.Linear(28, 256)

        # 交叉注意力层
        self.cross_attention_grammar_to_semantic = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.2, batch_first=True)  #batch_first=True
        self.cross_attention_semantic_to_grammar = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.2, batch_first=True)  #batch_first=True
        
        self.Classifier = nn.Sequential(
            nn.Linear(256 * 2, 64),  # 拼接后的维度变为 512
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, grammar_feature, semantic_feature):

        # 通过 BiLSTM 处理语义特征
        semantic_feature_fusion = self.BiLSTM(semantic_feature)
        # 通过自注意力模块处理语义特征
        semantic_feature_all = self.SelfAttention(semantic_feature_fusion)


        # 投影 grammar_feature 到 256 维
        grammar_feature_projected = self.grammar_projection(grammar_feature)  # (batch, 256)

        # 添加 batch 维度以适应注意力层 (batch, 1, 256)
        grammar_feature_projected = grammar_feature_projected.unsqueeze(1)
        semantic_feature_all = semantic_feature_all.unsqueeze(1)
        

        # Grammar → Semantic Attention
        attn_output_g2s, _ = self.cross_attention_grammar_to_semantic(grammar_feature_projected, 
                                                                      semantic_feature_all, 
                                                                      semantic_feature_all)
        attn_output_g2s = attn_output_g2s.squeeze(1)  # (batch, 256)

        # Semantic → Grammar Attention
        attn_output_s2g, _ = self.cross_attention_semantic_to_grammar(semantic_feature_all, 
                                                                      grammar_feature_projected, 
                                                                      grammar_feature_projected)
        attn_output_s2g = attn_output_s2g.squeeze(1)  # (batch, 256)


        # 融合两种特征
        combined_feature = torch.cat((attn_output_g2s, attn_output_s2g), dim=1)  # (batch, 512)


        # 通过分类器
        output = self.Classifier(combined_feature)

    
        return output



