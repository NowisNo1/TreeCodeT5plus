import json
import os
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, T5Config, get_linear_schedule_with_warmup, T5ForConditionalGeneration
from gson_reader import read_gson, CallTreeDataset, read_gsons
import tqdm


batch_size = 16
num_epochs = 20
learning_rate = 2e-5
warmup_steps = 500
output_dir = "../saved_model_origin"  # 模型保存路径

seq_length = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model_path = "../CodeT5p-220m"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

train_data, test_data = read_gsons(["data_filter_train.json"]), read_gsons(["data_filter_test.json"])

train_dataset = CallTreeDataset(train_data, tokenizer=tokenizer, max_length=seq_length)
test_dataset = CallTreeDataset(test_data, tokenizer=tokenizer, max_length=seq_length)


def collate_graphs(x, tokenizer, max_length):
    batch_size = len(x)
    sample_tensors = []
    labels = []

    for idx in range(batch_size):
        item = x[idx]
        root = item["root"]
        method_full_name = root["method_full_name"]
        method_body = root["method_body"]
        masked_root_method, label = mask_method_method(method_full_name, method_body)

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
            "attention_mask": attention_masks}


def mask_method_method(method_full_name, method_body, mask="<mask>"):
    pos = method_full_name.rfind('.')
    name = method_full_name
    if pos != -1:
        name = method_full_name[pos + 1:]

    return method_body.replace(name, mask).strip(), name


dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,  # 批次大小
    shuffle=True,  # 打乱数据
    collate_fn=lambda x: collate_graphs(x, tokenizer, seq_length)
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,  # 批次大小
    shuffle=False,  # 打乱数据
    collate_fn=lambda x: collate_graphs(x, tokenizer, seq_length)
)
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(dataloader) * num_epochs

print(f"total_steps: {total_steps}")
time.sleep(1)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# --------------------------
# 4. 训练循环
# --------------------------
model.train()  # 切换到训练模式

for epoch in range(num_epochs):
    total_loss = 0.0
    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for batch in progress_bar:
        # 批量数据移至设备
        input_ids = batch["graph"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # 重置梯度
        model.zero_grad()
        # 前向传播（计算损失）
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,  # 传入labels计算损失
        )
        loss = outputs.loss  # Seq2Seq模型的loss已在outputs中

        # 反向传播和参数更新
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 累计损失
        total_loss += loss.item()
        progress_bar.set_postfix({"Batch Loss": loss.item()})

    # 计算 epoch 平均损失
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")
    time.sleep(0.2)

    # 保存每个epoch的模型（可选）
    if True:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(f"{output_dir}/epoch_{epoch + 1}", safe_serialization=False)
        tokenizer.save_pretrained(f"{output_dir}/epoch_{epoch + 1}", safe_serialization=False)

print("训练完成！")
