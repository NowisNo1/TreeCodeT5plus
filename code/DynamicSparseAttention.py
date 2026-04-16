import torch
import torch.nn as nn
from transformers.models.t5.modeling_t5 import T5Attention

# 全局最大尺寸
MAX_N_CHILDREN = 10  # 最大子方法数量
MAX_SEQ_LEN = 512  # 最大序列长度，匹配 CodeT5p 默认


def get_nonzero_segments(inputs):

    # 如果没有非零元素，直接返回空结果
    if not inputs.any():
        return [], 0

    # 获取非零元素的索引
    nonzero_indices = inputs.nonzero(as_tuple=True)[-1]

    # 计算相邻索引的差值，用于检测是否连续
    diffs = nonzero_indices[1:] - nonzero_indices[:-1]

    # 找到差值大于1的位置（非连续点）
    split_points = (diffs > 1).nonzero(as_tuple=True)[0] + 1  # +1是为了得到分割点的正确位置

    # 构建分割索引列表（包括首尾）
    split_indices = torch.cat([
        torch.tensor([0]),
        split_points,
        torch.tensor([len(nonzero_indices)])
    ])

    # 计算每段的起始和结束索引
    segments = []
    for i in range(len(split_indices) - 1):
        start_idx = split_indices[i]
        end_idx = split_indices[i + 1] - 1  # 最后一个元素的索引

        # 转换为原始张量中的实际索引
        segment_start = nonzero_indices[start_idx].item()
        segment_end = nonzero_indices[end_idx].item()
        segment_length = segment_end - segment_start + 1

        segments.append({
            'start': segment_start,
            'end': segment_end,
            'length': segment_length
        })

    return segments, len(segments)


class DynamicSparseAttention(T5Attention):
    def __init__(self, config):
        super().__init__(config)
        self.last_attention_weights = None

    def forward(self, hidden_states, mask=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        if mask is not None:
            # 打印 mask 以调试
            print(f"Received mask shape: {hidden_states.shape}")  # 打印第一批第一头的值
            # 转换为布尔掩码，忽略 -inf
            child_mask = (mask != float('-inf')).float()  # -inf 表示忽略
            weighted_states = hidden_states.clone()

            for b in range(batch_size):
                childs, n_children = get_nonzero_segments(child_mask[b])
                print(f"{childs}, {n_children}")
                if n_children > 0:
                    for seg in range(n_children):
                        start_idx = childs[seg]["start"]
                        end_idx = childs[seg]["end"]
                        n_child_tokens = end_idx - start_idx + 1
                        # 动态生成注意力权重，填充多余部分
                        attention_weights = torch.zeros(MAX_SEQ_LEN)
                        if n_children > 0:
                            attention_weights[:n_children] = torch.ones(n_children) / n_children
                        attention_weights = torch.softmax(attention_weights, dim=-1).unsqueeze(0)  # [1, MAX_N_CHILDREN]

                        # 调整权重维度以匹配子方法 token 数
                        if n_child_tokens > n_children:
                            attention_weights = \
                                attention_weights.repeat(1, n_child_tokens // n_children + 1)[:, :n_child_tokens]
                        elif n_child_tokens < MAX_N_CHILDREN:
                            attention_weights = attention_weights[:, :n_child_tokens]
                            # 记录权重

                        self.last_attention_weights = attention_weights

                        # 加权子方法嵌入
                        weighted_states[b, start_idx:end_idx + 1, :] *= attention_weights

            output = super().forward(weighted_states, mask=mask, **kwargs)
            return output
        else:
            output = super().forward(hidden_states, mask=mask, **kwargs)
            return output
