import os
import re
import time
from types import SimpleNamespace
from datasets import Dataset
import torch
from functools import partial

from onnxruntime.quantization import quantize_dynamic, QuantType
from optimum.onnxruntime import (
    ORTModelForSeq2SeqLM,
    ORTQuantizer,
    ORTOptimizer
)
from optimum.onnxruntime.configuration import (
    AutoQuantizationConfig,
    AutoCalibrationConfig,
    OptimizationConfig
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration

from T5WithTreeModel import T6ForConditionalGeneration
from gson_reader import read_gson, CallTreeDataset, CalibrationDataset
import tqdm
from torch.amp import autocast


n_samples = 50
batch_size = 24
seq_length = 128
label_length = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
root_path = "../CodeT5p_no_mask/"
model_path = root_path + "epoch_1"
tokenizer = AutoTokenizer.from_pretrained(model_path)

data = read_gson("test_d_2_b_128_l_16_mask_recursive_no_test.json")

#
data, _calibration = data[n_samples:], data[:n_samples]

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
        return 0.0, 1

    return intersection_size, pred_size


def recall_mnr(true_tokens, pred_tokens):
    true_filtered = clean_whitespace_tokens(true_tokens)
    pred_filtered = clean_whitespace_tokens(pred_tokens)

    true_set = set(true_filtered)
    pred_set = set(pred_filtered)
    intersection_size = len(true_set & pred_set)
    true_size = len(true_set)

    if true_size == 0:
        return 0.0, 1

    return intersection_size, true_size


def collate_graphs(x, tokenizer, max_length, max_label_length):

    node_to_sample_ptr = [0]
    edge_to_sample_ptr = [0]
    current = 0
    current_edge = 0
    batch_size = len(x)
    input_tensors = []
    input_attn_masks = []
    graph_tensors = []
    graph_attn_masks = []
    flat_gat_edges = []
    origin_labels = []
    sub_method_masks = []

    for idx in range(batch_size):
        item = x[idx]
        root = item["root"]
        method_full_name = root["method_full_name"]
        method_body = root["method_body"]
        pos = method_full_name.rfind('.')
        pos_end = method_full_name.rfind("(")
        label = method_full_name
        if pos != -1 and pos_end != -1:
            label = method_full_name[pos + 1: pos_end]
        masked_root_method = method_body


        nodes, edges, num_son, childs = parse(root, 0, [[masked_root_method, 0]], [], 0, 1, 0, [])

        nodes = [item[0] for item in nodes]
        nodes = nodes[1:]
        # pattern = r"<extra_id_\d+>"
        # masked_root_method = re.sub(pattern, "", masked_root_method)
        # pattern = r"<extra_id_\d+>"
        # for i in range(len(root["children"])):
        #     # child = root["children"][i]
        #     # sub_full_name = child["method_full_name"]
        #     # sub_pos = sub_full_name.rfind('.')
        #     # sub_pos_end = sub_full_name.rfind("(")
        #     # sub_name = method_full_name
        #     # if sub_pos != -1 and sub_pos_end != -1:
        #     #     sub_name = sub_full_name[sub_pos + 1: sub_pos_end]
        #     # nodes[i] = re.sub(pattern, sub_name, nodes[i])
        #     nodes[i] = re.sub(pattern, "", nodes[i])

        current += len(nodes)
        node_to_sample_ptr.append(current)

        input_tensor = tokenizer(
            masked_root_method,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        mask = input_tensor["input_ids"] >= 32000
        indices = torch.nonzero(mask[0], as_tuple=True)[0]
        sub_method_mask = torch.zeros(seq_length, dtype=torch.bool)
        sub_method_mask[indices] = True
        sub_method_masks.append(sub_method_mask)

        input_tensors.extend(input_tensor["input_ids"])
        input_attn_masks.extend(input_tensor["attention_mask"])

        if len(nodes) != 0:
            graph_tensor = tokenizer(
                nodes,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )

            graph_tensors.extend(graph_tensor["input_ids"])
            graph_attn_masks.extend(graph_tensor["attention_mask"])

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

        origin_labels.append(label)


    input_tensors = torch.stack(input_tensors, dim=0)
    input_attn_masks = torch.stack(input_attn_masks, dim=0)

    if len(graph_tensors) > 0:
        graph_tensors = torch.stack(graph_tensors, dim=0)
        graph_attn_masks = torch.stack(graph_attn_masks, dim=0)
    else:
        graph_tensors = None
        graph_attn_masks = None

    sub_method_masks = torch.stack(sub_method_masks, dim=0)

    node_to_sample_ptr = torch.tensor(node_to_sample_ptr, dtype=torch.long)
    edge_to_sample_ptr = torch.tensor(edge_to_sample_ptr, dtype=torch.long)
    gat_edges = flat_gat_edges

    return {"input_tensors": input_tensors,
            "input_attn_masks": input_attn_masks, "graph_tensors": graph_tensors,
            "graph_attn_masks": graph_attn_masks, "sub_method_masks": sub_method_masks,
            "gat_edges": gat_edges, "node_to_sample_ptr": node_to_sample_ptr, "edge_to_sample_ptr": edge_to_sample_ptr, "origin": origin_labels}


def parse(root, current_idx, nodes, edges, depth, threshold, num_son, childs):
    if "children" not in root.keys() or depth == threshold:
        return nodes, edges, num_son, childs
    children = root["children"]
    if len(children) > 0:
        for i in range(len(children)):
            method = children[i]["method_body"]
            num_son += 1
            nodes.append([method.strip(), num_son])
            childs.append(children[i])
            edges.append([current_idx, num_son])

            _, _, num_son, childs = parse(children[i], num_son, nodes, edges, depth + 1, threshold, num_son, childs)

    return nodes, edges, num_son, childs


def processed_dataset(x, tokenizer, max_length, label_length, model):

    node_to_sample_ptr = [0]
    edge_to_sample_ptr = [0]
    current = 0
    current_edge = 0
    input_tensors = []
    input_attn_masks = []
    graph_tensors = []
    graph_attn_masks = []
    flat_gat_edges = []
    origin_labels = []
    sub_method_masks = []
    labels = []

    root = x
    method_full_name = root["method_full_name"]
    method_body = root["method_body"]
    pos = method_full_name.rfind('.')
    pos_end = method_full_name.rfind("(")
    label = method_full_name
    if pos != -1 and pos_end != -1:
        label = method_full_name[pos + 1: pos_end]
    masked_root_method = method_body
    label_tensor = tokenizer(
        label,
        truncation=True,
        padding="max_length",
        max_length=label_length,
        return_tensors="pt"
    ).input_ids
    labels.append(label_tensor)

    labels = torch.stack(labels, dim=0).squeeze(1)
    nodes, edges, num_son, childs = parse(root, 0, [[masked_root_method, 0]], [], 0, 1, 0, [])

    nodes = [item[0] for item in nodes]
    nodes = nodes[1:]
    # pattern = r"<extra_id_\d+>"
    # masked_root_method = re.sub(pattern, "", masked_root_method)
    # pattern = r"<extra_id_\d+>"
    # for i in range(len(root["children"])):
    #     # child = root["children"][i]
    #     # sub_full_name = child["method_full_name"]
    #     # sub_pos = sub_full_name.rfind('.')
    #     # sub_pos_end = sub_full_name.rfind("(")
    #     # sub_name = method_full_name
    #     # if sub_pos != -1 and sub_pos_end != -1:
    #     #     sub_name = sub_full_name[sub_pos + 1: sub_pos_end]
    #     # nodes[i] = re.sub(pattern, sub_name, nodes[i])
    #     nodes[i] = re.sub(pattern, "", nodes[i])

    current += len(nodes)
    node_to_sample_ptr.append(current)

    input_tensor = tokenizer(
        masked_root_method,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

    mask = input_tensor["input_ids"] >= 32000
    indices = torch.nonzero(mask[0], as_tuple=True)[0]
    sub_method_mask = torch.zeros(seq_length, dtype=torch.bool)
    sub_method_mask[indices] = True
    sub_method_masks.append(sub_method_mask)

    input_tensors.extend(input_tensor["input_ids"])
    input_attn_masks.extend(input_tensor["attention_mask"])

    if len(nodes) != 0:
        graph_tensor = tokenizer(
            nodes,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )

        graph_tensors.extend(graph_tensor["input_ids"])
        graph_attn_masks.extend(graph_tensor["attention_mask"])

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

    origin_labels.append(label)


    input_tensors = torch.stack(input_tensors, dim=0)
    input_attn_masks = torch.stack(input_attn_masks, dim=0)

    if len(graph_tensors) > 0:
        graph_tensors = torch.stack(graph_tensors, dim=0)
        graph_attn_masks = torch.stack(graph_attn_masks, dim=0)
    else:
        graph_tensors = None
        graph_attn_masks = None

    sub_method_masks = torch.stack(sub_method_masks, dim=0)

    node_to_sample_ptr = torch.tensor(node_to_sample_ptr, dtype=torch.long)
    edge_to_sample_ptr = torch.tensor(edge_to_sample_ptr, dtype=torch.long)
    gat_edges = flat_gat_edges


    decoder_outputs = model(
        input_ids=input_tensors.to(device),
        attention_mask=input_attn_masks.to(device),
        labels=labels.to(device),
        # sub_method_masks=sub_method_masks.to(device),
        # gat_edges=gat_edges,
        # node_to_sample_ptr=node_to_sample_ptr.to(device),
        # edge_to_sample_ptr=edge_to_sample_ptr.to(device),
        # graph_input_ids=graph_tensors.to(device),
        # graph_attention_mask=graph_attn_masks.to(device),
        use_cache=True
    )
    tmp = decoder_outputs["past_key_values"]

    ret = {
        "encoder_attention_mask": input_attn_masks.squeeze(0),
        "encoder_hidden_states": decoder_outputs.encoder_last_hidden_state.squeeze(0),
        "input_ids": input_tensors.squeeze(0), "label": labels,
        "attention_mask": input_attn_masks.squeeze(0), "graph_ids": graph_tensors,
        "graph_attention_mask": graph_attn_masks, "sub_method_masks": sub_method_masks,
        "gat_edges": gat_edges, "node_to_sample_ptr": node_to_sample_ptr,
        "edge_to_sample_ptr": edge_to_sample_ptr, "origin": origin_labels
    }

    for _ in range(0, 12):
        ret[f'past_key_values.{_}.decoder.key'] = tmp[_][3].squeeze(0)
        ret[f'past_key_values.{_}.decoder.value'] = tmp[_][2].squeeze(0)
        ret[f'past_key_values.{_}.encoder.key'] = tmp[_][1].squeeze(0)
        ret[f'past_key_values.{_}.encoder.value'] = tmp[_][0].squeeze(0)

    return ret


_calibration = [data["root"] for data in _calibration]


paths = os.listdir(root_path)
num_epochs = 15

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,  # 批次大小
    collate_fn=lambda x: collate_graphs(x, tokenizer, seq_length, label_length)
)


for i in range(1, num_epochs + 1):
    if i < 1:
        continue
    path = "epoch_" + str(i)

    model_path = root_path + path
    data_type = torch.bfloat16
    with autocast(device_type=device.type, dtype=data_type):
        if True:
            tmp = "../CodeT5p_ptq_int8_model_tmp/" + path
            quantization_dir = "../CodeT5p_ptq_int8_model/" + path
            ort_model = ORTModelForSeq2SeqLM.from_pretrained(
                model_path,
                export=True,
            ).to(device)
            model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
            calibration_dataset = Dataset.from_list(_calibration).map(processed_dataset, batched=False, fn_kwargs={
                "tokenizer": tokenizer,
                "max_length": seq_length,
                "label_length": label_length,
                "model": model
            })
            ort_model.save_pretrained(tmp, safe_serialization=False)
            onnx_files = [
                "encoder_model.onnx",
                "decoder_model.onnx",
                "decoder_with_past_model.onnx"
            ]
            select_columns = [
                ["input_ids", "attention_mask"],
                ["input_ids", "encoder_attention_mask", "encoder_hidden_states"],
                ["input_ids", "encoder_attention_mask", 'past_key_values.0.decoder.key',
                 'past_key_values.0.decoder.value', 'past_key_values.0.encoder.key',
                 'past_key_values.0.encoder.value', 'past_key_values.1.decoder.key',
                 'past_key_values.1.decoder.value', 'past_key_values.1.encoder.key',
                 'past_key_values.1.encoder.value', 'past_key_values.2.decoder.key',
                 'past_key_values.2.decoder.value', 'past_key_values.2.encoder.key',
                 'past_key_values.2.encoder.value', 'past_key_values.3.decoder.key',
                 'past_key_values.3.decoder.value', 'past_key_values.3.encoder.key',
                 'past_key_values.3.encoder.value', 'past_key_values.4.decoder.key',
                 'past_key_values.4.decoder.value', 'past_key_values.4.encoder.key',
                 'past_key_values.4.encoder.value', 'past_key_values.5.decoder.key',
                 'past_key_values.5.decoder.value', 'past_key_values.5.encoder.key',
                 'past_key_values.5.encoder.value', 'past_key_values.6.decoder.key',
                 'past_key_values.6.decoder.value', 'past_key_values.6.encoder.key',
                 'past_key_values.6.encoder.value', 'past_key_values.7.decoder.key',
                 'past_key_values.7.decoder.value', 'past_key_values.7.encoder.key',
                 'past_key_values.7.encoder.value', 'past_key_values.8.decoder.key',
                 'past_key_values.8.decoder.value', 'past_key_values.8.encoder.key',
                 'past_key_values.8.encoder.value', 'past_key_values.9.decoder.key',
                 'past_key_values.9.decoder.value', 'past_key_values.9.encoder.key',
                 'past_key_values.9.encoder.value', 'past_key_values.10.decoder.key',
                 'past_key_values.10.decoder.value', 'past_key_values.10.encoder.key',
                 'past_key_values.10.encoder.value', 'past_key_values.11.decoder.key',
                 'past_key_values.11.decoder.value', 'past_key_values.11.encoder.key',
                 'past_key_values.11.encoder.value'],
            ]
            q_config = AutoQuantizationConfig.arm64(
                is_static=True,
                per_channel=True,
                operators_to_quantize=["MatMul", "Gemm"],
            )
            q_config.weights_dtype = QuantType.QInt8
            _idx = 0
            for file, col in zip(onnx_files, select_columns):
                input_path = os.path.join(tmp, file)
                output_path = os.path.join(quantization_dir, file)
                quantizer = ORTQuantizer.from_pretrained(tmp, file_name=file)
                tmp_dataset = calibration_dataset.select_columns(col)
                calibration_config = AutoCalibrationConfig.minmax(tmp_dataset)
                ranges = quantizer.fit(
                    dataset=tmp_dataset,
                    calibration_config=calibration_config,
                    operators_to_quantize=q_config.operators_to_quantize
                )

                # ranges = collect_calibration_ranges(ort_model)
                quantizer.quantize(
                    quantization_config=q_config,
                    save_dir=quantization_dir,
                    calibration_tensors_range=ranges,
                )
            model = ORTModelForSeq2SeqLM.from_pretrained(quantization_dir, provider="CUDAExecutionProvider").to(device)
        # model = ORTModelForSeq2SeqLM.from_pretrained(model_path, provider="CUDAExecutionProvider").to(device)
        # except Exception as e:
        #     print(e)
        #     continue

        print(model_path)
        time.sleep(0.1)
        success = 0
        totol = 0
        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {path}")
        total_inter_pre_size = 0
        total_pred_size = 0
        total_inter_rec_size = 0
        total_true_size = 0
        for batch in progress_bar:
            idx = 0
            graph_ids = batch["graph_tensors"]
            input_ids = batch["input_tensors"].to(device)
            graph_attention_mask = batch["graph_attn_masks"]
            attention_mask = batch["input_attn_masks"].to(device)
            sub_method_masks = batch["sub_method_masks"].to(device)
            gat_edges = [edge.to(device) for edge in batch["gat_edges"]]
            node_to_sample_ptr = batch["node_to_sample_ptr"].to(device)
            edge_to_sample_ptr = batch["edge_to_sample_ptr"].to(device)

            if graph_ids is not None:
                graph_ids = graph_ids.to(device)
                graph_attention_mask = graph_attention_mask.to(device)
            else:
                gat_edges = None
            node_to_sample_ptr = torch.tensor([i for i in range(input_ids.shape[0] + 1)], device=device)
            gat_edges = None
            encoder_last_hidden_state = None
            batch_size = input_ids.size(0)
            decoder_start_token_id = tokenizer.pad_token_id
            generated_ids = torch.full(
                (batch_size, 1),
                decoder_start_token_id,
                device=device
            )

            end_token_id = tokenizer.eos_token_id

            for _ in range(label_length):
                if encoder_last_hidden_state is None:
                    with torch.no_grad():
                        decoder_outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=generated_ids,
                            decoder_attention_mask=torch.ones_like(generated_ids),
                            # sub_method_masks=sub_method_masks,
                            # gat_edges=gat_edges,
                            # node_to_sample_ptr=node_to_sample_ptr,
                            # edge_to_sample_ptr=edge_to_sample_ptr,
                            # graph_input_ids=graph_ids,
                            # graph_attention_mask=graph_attention_mask
                        )
                        encoder_last_hidden_state = decoder_outputs.encoder_last_hidden_state
                        encoder_attention_mask = attention_mask

                else:
                    with torch.no_grad():
                        decoder_outputs = model(
                            input_ids=None,
                            attention_mask=encoder_attention_mask,
                            encoder_outputs=SimpleNamespace(last_hidden_state=encoder_last_hidden_state),
                            decoder_input_ids=generated_ids,
                            decoder_attention_mask=torch.ones_like(generated_ids),
                        )
                next_token_logits = decoder_outputs.logits[:, -1, :]
                next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
                if (next_token_ids == end_token_id).all().item():
                    break

            generated_texts = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            for txt in generated_texts:
                true_tokens = camel_case_tokenize(batch['origin'][idx])
                pred_tokens = camel_case_tokenize(txt.strip())
                inter_pre_size, pred_size = precision_mnr(true_tokens, pred_tokens)
                inter_rec_size, true_size = recall_mnr(true_tokens, pred_tokens)
                if len(batch['origin'][idx]) > 0 and len(true_tokens) == 0:
                    print("?")
                total_inter_pre_size += inter_pre_size
                total_pred_size += pred_size
                total_inter_rec_size += inter_rec_size
                total_true_size += true_size

                if txt.strip() == batch['origin'][idx]:
                    success += 1

                idx += 1
                totol += 1

            _pre = total_inter_pre_size / total_pred_size
            _rec = total_inter_rec_size / total_true_size
            ff1 = 0
            if total_inter_pre_size + total_inter_rec_size > 0:
                ff1 = 2 * _pre * _rec / (_pre + _rec)
            try:
                progress_bar \
                    .set_postfix({ ##"node size": f"{graph_ids.shape[0] if graph_ids is not None else 0}",
                                  "accuracy": f"{(1.0 * success / totol):.5f}",
                                  "precision": f"{_pre:.5f}",
                                  "recall": f"{_rec:.5f}",
                                  "f-score": f"{ff1:.5f}"})
            finally:
                continue
