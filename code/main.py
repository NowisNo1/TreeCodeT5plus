# import torch
# from transformers import AutoTokenizer, T5ForConditionalGeneration
# from DynamicSparseAttention import DynamicSparseAttention
#
# model_path = "models/CodeT5p-220m"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = T5ForConditionalGeneration.from_pretrained(model_path)
#
# # 全局最大尺寸
# MAX_N_CHILDREN = 10  # 最大子方法数量
# MAX_SEQ_LEN = 512   # 最大序列长度，匹配 CodeT5p 默认
# MAX_DECODER_LEN = 128  # 最大解码器输出长度，可根据需求调整
#
#
# # 统一尺寸的输入预处理
# def encode_tree_input(parent, children, child_names,
#                       max_n_children=MAX_N_CHILDREN,
#                       max_seq_len=MAX_SEQ_LEN,
#                       max_decoder_len=MAX_DECODER_LEN):
#
#     prompt = f"<extra_id_0> {parent} <extra_id_1>"
#     for child, name in zip(children, child_names):
#         prompt += f" <extra_id_2> {child} <extra_id_3> <extra_id_4> {name} <extra_id_5>"
#     inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=max_seq_len, truncation=True)
#
#     # 生成 child_mask 并填充
#     child_mask = torch.full_like(inputs["input_ids"], float('-inf'), dtype=torch.float)
#     start_indices = (inputs["input_ids"] == tokenizer.convert_tokens_to_ids("<extra_id_2>")).nonzero(as_tuple=True)
#     end_indices = (inputs["input_ids"] == tokenizer.convert_tokens_to_ids("<extra_id_3>")).nonzero(as_tuple=True)
#
#     for b in range(inputs["input_ids"].size(0)):
#         for i in range(min(len(children), max_n_children)):
#             if i < len(start_indices[0]):
#                 start_idx = start_indices[1][start_indices[0] == b][i].item()
#                 end_idx = end_indices[1][end_indices[0] == b][i].item()
#                 child_mask[b, start_idx:end_idx + 1] = 0  # 子函数位置设为 0
#
#     inputs["attention_mask"] = child_mask
#
#     decoder_input_ids = torch.ones((inputs["input_ids"].size(0), max_decoder_len),
#                                    dtype=torch.long) * tokenizer.pad_token_id
#     decoder_input_ids[:, 0] = tokenizer.bos_token_id
#     inputs["decoder_input_ids"] = decoder_input_ids
#
#     return inputs
#
#
# # 集成注意力机制
# model.encoder.block[0].layer[0].SelfAttention = DynamicSparseAttention(model.config)
#
# # 示例运行
# # 案例 1：2 个子方法
# parent1 = "def parent_method():\n    pass"
# children1 = ["def child1():\n    return 1", "def child2():\n    return 2"]
# child_names1 = ["child1", "child2"]
#
# # 案例 2：3 个子方法
# parent2 = "def parent_method():\n    pass"
# children2 = ["def child1():\n    return 1", "def child2():\n    return 2", "def child3():\n    return 3"]
# child_names2 = ["child1", "child2", "child3"]
#
# # 批量处理
# batch_size = 2
# inputs1 = encode_tree_input(parent1, children1, child_names1)
# inputs2 = encode_tree_input(parent2, children2, child_names2)
# inputs = {"input_ids": torch.cat([inputs1["input_ids"], inputs2["input_ids"]], dim=0),
#           "attention_mask": torch.cat([inputs1["attention_mask"], inputs2["attention_mask"]], dim=0),
#           "decoder_input_ids": torch.cat([inputs1["decoder_input_ids"], inputs2["decoder_input_ids"]], dim=0)}
#
#
# attention = model.encoder.block[0].layer[0].SelfAttention
#
# print(inputs1)
# # outputs = model(**inputs)
# # print(f"Attention Weights: {attention.last_attention_weights.detach().numpy() if attention.last_attention_weights is not None else 'None'}")
#
res = {}
res[1] = 1
print(res)