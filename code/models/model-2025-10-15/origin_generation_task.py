import os
import re
import time
# 0.4943
# 0.5398275222551929
"""
Epoch epoch_15: 100%|██████████| 899/899 [2:04:33<00:00,  8.31s/it, accuracy=0.539828, precision=0.679068, recall=0.676824, f-score=0.672557, f-score2=0.677944]
"""
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5Config, get_linear_schedule_with_warmup, T5ForConditionalGeneration
from gson_reader import read_gson, CallTreeDataset
import tqdm


batch_size = 24

seq_length = 128
max_length = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model_path = "../saved_model_origin/epoch_1"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# for param in model.named_parameters():
#     print(param)

data = read_gson("data_filter_test.json")

dataset = CallTreeDataset(data, tokenizer=tokenizer, max_length=seq_length)


def camel_case_tokenize(text):
    return re.sub('([a-z])([A-Z])', r'\1 \2', text).split()


def clean_whitespace_tokens(tokens):
    cleaned_tokens = []
    for token in tokens:
        stripped_token = token.strip()
        if not stripped_token or stripped_token.isspace():
            continue
        cleaned_tokens.append(stripped_token)
    return cleaned_tokens


def filter_special_tokens(tokens, tokenizer):
    tokens = clean_whitespace_tokens(tokens)
    special_tokens = set()
    if hasattr(tokenizer, 'all_special_tokens'):
        special_tokens.update(tokenizer.all_special_tokens)

    filtered_tokens = [
        token for token in tokens
        if token not in special_tokens
        and not (isinstance(token, str) and token.startswith(('##', '<|', '[|')))
    ]
    return filtered_tokens


def precision_mnr(true_tokens, pred_tokens):
    true_filtered = clean_whitespace_tokens(true_tokens)
    pred_filtered = clean_whitespace_tokens(pred_tokens)

    true_set = set(true_filtered)
    pred_set = set(pred_filtered)
    intersection_size = len(true_set & pred_set)
    pred_size = len(pred_set)

    if pred_size == 0:
        return 0.0

    return intersection_size / pred_size


def recall_mnr(true_tokens, pred_tokens):
    true_filtered = clean_whitespace_tokens(true_tokens)
    pred_filtered = clean_whitespace_tokens(pred_tokens)

    true_set = set(true_filtered)
    pred_set = set(pred_filtered)
    intersection_size = len(true_set & pred_set)
    true_size = len(true_set)

    if true_size == 0:
        return 0.0

    return intersection_size / true_size



def collate_graphs(x, tokenizer, max_length):
    batch_size = len(x)
    sample_tensors = []
    labels = []
    origin_labels = []
    for idx in range(batch_size):
        item = x[idx]
        root = item["root"]
        method_full_name = root["method_full_name"]
        method_body = root["method_body"]
        masked_root_method, label = mask_method_method(method_full_name, method_body)
        origin_labels.append(label)
        # 标签转换为张量
        label_tensor = tokenizer(
            label,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).input_ids
        labels.append(label_tensor)
        sample_tensors.append(masked_root_method)

    output = tokenizer(
        sample_tensors,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    sample_tensors, attention_masks = output["input_ids"], output["attention_mask"]
    labels = torch.stack(labels, dim=0).squeeze(1)

    return {"graph": sample_tensors, "label": labels,
            "attention_mask": attention_masks, "origin": origin_labels}


def mask_method_method(method_full_name, method_body, mask="<mask>"):
    pos = method_full_name.rfind('.')
    name = method_full_name
    if pos != -1:
        name = method_full_name[pos + 1:]

    return method_body.replace(name, mask).strip(), name


paths = os.listdir("../saved_model_origin")
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,  # 批次大小
    collate_fn=lambda x: collate_graphs(x, tokenizer, seq_length)
)
for path in paths:
    model_path = "../saved_model_origin/" + path
    if path == "log":
        continue
    print(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

    time.sleep(1)

    success = 0
    totol = 0
    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {path}")
    total_precision = 0
    total_recall = 0
    total_f_score = 0
    for batch in progress_bar:
        idx = 0
        input_ids = batch["graph"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,  # 传入labels计算损失
            )
        encoder_last_hidden_state = outputs.encoder_last_hidden_state  # 形状：(batch_size, encoder_seq_len, hidden_size)
        # 编码器的 attention mask（避免关注padding）
        encoder_attention_mask = attention_mask

        # 3. 初始化解码器输入（通常以起始token开始，如</s>或模型的decoder_start_token_id）
        batch_size = encoder_last_hidden_state.size(0)
        decoder_start_token_id = tokenizer.pad_token_id
        generated_ids = torch.full(
            (batch_size, 1),  # 初始形状：(batch_size, 1)
            decoder_start_token_id,
            device=device  # 与模型同设备
        )
        # 4. 迭代生成token（贪心搜索为例）
        end_token_id = tokenizer.eos_token_id  # 终止token（如</s>）

        for _ in range(max_length):
            # 解码器前向传播（需要传入编码器输出和mask）
            with torch.no_grad():
                decoder_outputs = model(
                    input_ids=None,  # 编码器输入已通过encoder_outputs传入，这里不重复传
                    attention_mask=encoder_attention_mask,
                    encoder_outputs=(encoder_last_hidden_state,),  # 传入编码器输出
                    decoder_input_ids=generated_ids,  # 当前生成的解码器序列
                    decoder_attention_mask=torch.ones_like(generated_ids),  # 解码器mask（全为1，无padding）
                )

            # 获取解码器最后一个位置的logits
            next_token_logits = decoder_outputs.logits[:, -1, :]  # 形状：(batch_size, vocab_size)
            next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # 贪心选择

            # 拼接新生成的token
            generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)

            # 检查是否所有样本都生成了终止token，提前退出
            if (next_token_ids == end_token_id).all().item():
                break

        # 解码为文本
        generated_texts = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        for txt in generated_texts:
            true_tokens = camel_case_tokenize(batch['origin'][idx])
            pred_tokens = camel_case_tokenize(txt.strip())
            precision = precision_mnr(true_tokens, pred_tokens)
            recall = recall_mnr(true_tokens, pred_tokens)
            total_precision += precision
            total_recall += recall
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                total_f_score += f1
            if txt.strip() == batch['origin'][idx]:
                success += 1

            idx += 1
            totol += 1
        pre = 1.0 * total_precision
        recall_1 = 1.0 * total_recall
        progress_bar \
            .set_postfix({"accuracy": f"{(1.0 * success / totol):.6f}",
                          "precision": f"{(1.0 * total_precision / totol):.6f}",
                          "recall": f"{(1.0 * total_recall / totol):.6f}",
                          "f-score": f"{(1.0 * total_f_score / totol):.6f}",
                          "f-score2": f"{2 * pre * recall_1 / (pre + recall_1) / totol:.6f}"})

