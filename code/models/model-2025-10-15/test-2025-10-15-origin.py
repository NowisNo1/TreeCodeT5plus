import json
import os
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, T5Config, get_linear_schedule_with_warmup
from T5WithTreeModel import T6ForConditionalGeneration
from gson_reader import read_gson, CallTreeDataset, read_gsons
import tqdm


batch_size = 16
num_epochs = 20
learning_rate = 2e-5
warmup_steps = 500
output_dir = "../saved_model"  # 模型保存路径

seq_length = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model_path = "../CodeT5p-220m"
# trained_model_path = "../saved_model/epoch_1"
# model_path = trained_model_path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T6ForConditionalGeneration.from_pretrained(model_path).to(device)
model.init_params()
## for param in model.named_parameters():
##     print(param)
# prefix = "../../dataset/benchmark"
# paths = []
# for path in os.listdir(prefix):
#     paths.append(prefix + "/" + path)
#
# data = read_gsons(paths)
#
# dataset = CallTreeDataset(data, tokenizer=tokenizer, max_length=seq_length)
# train_size = 0.16 # int(0.8 * len(dataset))
# test_size = 0.04 # len(dataset) - train_size
#
# train_dataset, test_dataset, _ = random_split(dataset, [train_size, test_size, 0.8])
# with open("train.json", "w") as f:
#     tmp = []
#     for idx in train_dataset.indices:
#         tmp.append(data[idx])
#     json.dump(tmp, f)
#
# with open("test.json", "w") as f:
#     tmp = []
#     for idx in test_dataset.indices:
#         tmp.append(data[idx])
#     json.dump(tmp, f)
# exit(0)

train_data, test_data = read_gsons(["data_filter_train.json"]), read_gsons(["data_filter_test.json"])

train_dataset = CallTreeDataset(train_data, tokenizer=tokenizer, max_length=seq_length)
test_dataset = CallTreeDataset(test_data, tokenizer=tokenizer, max_length=seq_length)



def collate_graphs(x, tokenizer, max_length):

    node_to_sample_ptr = [0]
    edge_to_sample_ptr = [0]
    current = 0
    current_edge = 0
    batch_size = len(x)

    flat_tokens = []
    flat_gat_edges = []
    labels = []
    seq_idx2graph_idxs = []

    for idx in range(batch_size):
        item = x[idx]
        root = item["root"]
        method_full_name = root["method_full_name"]
        method_body = root["method_body"]
        masked_root_method, label = mask_method_method(method_full_name, method_body)
        nodes, edges, num_son = parse(root, 0, [[masked_root_method, 0]], [], 0, 3, 0)

        seq_idx2graph_idx = [item[1] for item in nodes]
        seq_idx2graph_idxs.extend(seq_idx2graph_idx)

        nodes = [item[0] for item in nodes]
        # print(num_son)
        current += len(nodes)
        node_to_sample_ptr.append(current)
        flat_tokens.extend(nodes)
        current_edge += len(edges)
        edge_to_sample_ptr.append(current_edge)

        gat_edge = torch.zeros(2, len(edges), dtype=torch.long)

        for i in range(len(edges)):
            gat_edge[0][i] = -1
            gat_edge[1][i] = -1
            if i < len(edges):
                gat_edge[0][i] = edges[i][0]
                gat_edge[1][i] = edges[i][1]

        flat_gat_edges.append(gat_edge)

        # 标签转换为张量
        label_tensor = tokenizer(
            label,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).input_ids
        labels.append(label_tensor)

    output = tokenizer(
        flat_tokens,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    sample_tensors, attention_masks = output["input_ids"], output["attention_mask"]
    node_to_sample_ptr = torch.tensor(node_to_sample_ptr, dtype=torch.long)
    edge_to_sample_ptr = torch.tensor(edge_to_sample_ptr, dtype=torch.long)
    seq_idx2graph_idxs = torch.tensor(seq_idx2graph_idxs, dtype=torch.long)
    labels = torch.stack(labels, dim=0).squeeze(1)
    gat_edges = flat_gat_edges

    return {"graph": sample_tensors, "label": labels,
            "attention_mask": attention_masks, "gat_edges": gat_edges,
            "node_to_sample_ptr": node_to_sample_ptr, "edge_to_sample_ptr": edge_to_sample_ptr,
            "seq_idx2graph_idx": seq_idx2graph_idxs}


def parse(root, current_idx, nodes, edges, depth, threshold, num_son):
    if "children" not in root.keys() or depth == threshold:
        return nodes, edges, num_son
    children = root["children"]
    if len(children) > 0:
        for i in range(len(children)):
            method = children[i]["method_body"]
            if method == "unResolve":
                continue
                # pos = children[i]["method_full_name"].rfind('.')
                # if pos == -1:
                #     method = children[i]["method_full_name"]
                # else:
                #     method = children[i]["method_full_name"][pos + 1:]

            num_son += 1
            nodes.append([method.strip(), num_son])

            edges.append([current_idx, num_son])

            _, _, num_son = parse(children[i], num_son, nodes, edges, depth + 1, threshold, num_son)

    return nodes, edges, num_son


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
        gat_edges = [edge.to(device) for edge in batch["gat_edges"]]
        node_to_sample_ptr = batch["node_to_sample_ptr"].to(device)
        edge_to_sample_ptr = batch["edge_to_sample_ptr"].to(device)
        seq_idx2graph_idx = batch["seq_idx2graph_idx"].to(device)

        # 重置梯度
        model.zero_grad()

        # 前向传播（计算损失）
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,  # 传入labels计算损失
            gat_edges=gat_edges,
            node_to_sample_ptr=node_to_sample_ptr,
            edge_to_sample_ptr=edge_to_sample_ptr,
            seq_idx2graph_idx=seq_idx2graph_idx
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
    # if epoch % 2 == 0 or epoch == num_epochs - 1:
    if True:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(f"{output_dir}/epoch_{epoch + 1}", safe_serialization=False)
        tokenizer.save_pretrained(f"{output_dir}/epoch_{epoch + 1}", safe_serialization=False)

print("训练完成！")
