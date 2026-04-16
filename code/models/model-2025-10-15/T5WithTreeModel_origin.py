import torch
import torch_geometric
from torch.nn import CrossEntropyLoss

from transformers import T5PreTrainedModel, GenerationMixin, add_start_docstrings, Cache, \
    GradientCheckpointingLayer, EncoderDecoderCache, DynamicCache, PretrainedConfig, T5Config
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import T5LayerSelfAttention, T5LayerCrossAttention, T5LayerFF, T5LayerNorm, \
    logger, T5Attention
from transformers.utils import is_torchdynamo_compiling, is_torch_flex_attn_available
from torch import nn
import copy
import warnings
from typing import Optional, Union

from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map

from CodeAwareGATConv import CodeAwareGATConv
from Fusion import ConcatMLPFusion
if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask

PARALLELIZE_DOCSTRING = r"""
    This is an experimental feature and is a subject to change at a moment's notice.

    Uses a device map to distribute attention modules of the model across several devices. If no device map is given,
    it will evenly distribute blocks across all devices.

    Args:
        device_map (`dict[int, list]`, *optional*):
            A dictionary that maps attention modules to devices. Note that the embedding module and LMHead are always
            automatically mapped to the first device (for esoteric reasons). That means that the first device should
            have fewer attention modules mapped to it than other devices. For reference, the t5 models have the
            following number of attention modules:

                - google-t5/t5-small: 6
                - google-t5/t5-base: 12
                - google-t5/t5-large: 24
                - google-t5/t5-3b: 24
                - google-t5/t5-11b: 24

    Example:

    ```python
    # Here is an example of a device map on a machine with 4 GPUs using google-t5/t5-3b, which has a total of 24 attention modules:
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)
    ```
"""
DEPARALLELIZE_DOCSTRING = r"""
    Moves the model to cpu from a model parallel state.

    Example:

    ```python
    # On a 4 GPU machine with google-t5/t5-3b:
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)  # Splits the model across several devices
    model.deparallelize()  # Put the model back on cpu and cleans memory by calling torch.cuda.empty_cache()
    ```
"""

torch.autograd.set_detect_anomaly(True)


def connected_components(edge_index, num_nodes=None):
    device = edge_index.device

    parent = torch.arange(num_nodes, device=device)
    rank = torch.zeros(num_nodes, device=device, dtype=torch.long)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        x_root = find(x)
        y_root = find(y)

        if x_root == y_root:
            return
        if rank[x_root] < rank[y_root]:
            parent[x_root] = y_root
        else:
            parent[y_root] = x_root
            if rank[x_root] == rank[y_root]:
                rank[x_root] += 1

    for i in range(edge_index.size(1)):
        union(edge_index[0, i], edge_index[1, i])

    components = torch.tensor([find(i) for i in range(num_nodes)], device=device)
    unique_roots = torch.unique(components)
    root_to_id = {root.item(): idx for idx, root in enumerate(unique_roots)}
    components = torch.tensor([root_to_id[root.item()] for root in components], device=device)

    return components


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, layer_idx: Optional[int] = None, fusion=None, gat_layer=None):
        super().__init__()

        self.SelfAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx
        )

        # TODO add gat_layer
        self.gat_layer = gat_layer
        self.fusion = fusion
        self.layer_idx = layer_idx
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            layer_head_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            cache_position=None,
            gat_edges=None,
            seq_idx2graph_idx=None
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        node_size, seq_length, hidden_size = normed_hidden_states.shape
        hidden_states_new = []
        attention_outputs = []
        remain_node_idx = []
        # if self.gat_layer is None:
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        if self.gat_layer is not None and gat_edges is not None and gat_edges.shape[1] != 0:

            gat_output, attn = self.gat_layer(hidden_states, gat_edges)
            if self.layer_idx % 2 == 1:
                # TODO Prune
                rate = (1 - self.layer_idx / 10)
                edge_index, alpha = attn
                alpha = alpha.reshape(-1)[:gat_edges.shape[1]].clone()
                total = alpha.numel()
                top_count = max(0, int(total * rate))
                _, indices = torch.sort(alpha, descending=True)
                indices, _ = torch.sort(indices[:top_count])

                device = hidden_states.device
                if len(indices) == 0:
                    gat_edges_new = torch.zeros(2, 0).to(device)
                else:
                    gat_edges_new = gat_edges.transpose(0, 1)[indices].transpose(0, 1).clone().to(device)

                max_graph_node = seq_idx2graph_idx.shape[0]
                components = connected_components(gat_edges_new, num_nodes=max_graph_node)

                target_component_idx = components[0]
                graph_connected_nodes = torch.where(components == target_component_idx)[0]

                is_in_connected = torch.zeros(max_graph_node, dtype=torch.bool, device=device)
                is_in_connected[graph_connected_nodes] = True

                valid_seq_indices = torch.where(is_in_connected[seq_idx2graph_idx])[0]
                valid_seq2graph = seq_idx2graph_idx[valid_seq_indices]

                graph_node_to_new_id = torch.full((max_graph_node, ), -1, dtype=torch.long, device=device)
                graph_node_to_new_id[graph_connected_nodes] = torch.arange(len(graph_connected_nodes), device=device)

                idx2idx = graph_node_to_new_id[valid_seq2graph].clone()

                gat_output_new = torch.zeros(graph_connected_nodes.shape[0],
                                             hidden_states.shape[1], hidden_states.shape[2]) \
                    .to(hidden_states.device)

                hidden_states_new = torch.zeros(graph_connected_nodes.shape[0],
                                                hidden_states.shape[1], hidden_states.shape[2]) \
                    .to(hidden_states.device)

                attention_output_new_first = torch.zeros(graph_connected_nodes.shape[0],
                                                         hidden_states.shape[1], hidden_states.shape[2]) \
                    .to(hidden_states.device)

                attention_output_new_other = torch.zeros(graph_connected_nodes.shape[0],
                                                         attention_output[1].shape[1],
                                                         attention_output[1].shape[2],
                                                         attention_output[1].shape[3]) \
                    .to(hidden_states.device)

                idx = 0
                for i in valid_seq_indices:
                    hidden_states_new[idx, :, :] = hidden_states[i, :, :]
                    attention_output_new_first[idx, :, :] = attention_output[0][i, :, :]
                    attention_output_new_other[idx, :, :, :] = attention_output[1][i, :, :, :]
                    gat_output_new[idx, :, :] = gat_output[i, :, :]
                    remain_node_idx.append(i)
                    idx += 1

                if gat_edges_new.size(1) > 0:
                    u_in = is_in_connected[gat_edges_new[0]]
                    v_in = is_in_connected[gat_edges_new[1]]
                    valid_edges_mask = u_in & v_in
                    valid_edges = gat_edges_new[:, valid_edges_mask]

                    gat_edges_new = graph_node_to_new_id[valid_edges]
                # TODO 更新 seq_idx2graph_idx 以及 gat_edges 和 hidden_states

                seq_idx2graph_idx = idx2idx.clone()
                gat_edges = gat_edges_new.clone()
                hidden_states = hidden_states_new.clone()
                remain_node_idx = torch.tensor(remain_node_idx)
                attention_output = (attention_output_new_first.clone(), attention_output_new_other.clone())
                gat_output = gat_output_new.clone()

                # print(attn[0])
            else:
                for i in range(seq_idx2graph_idx.shape[0]):
                    remain_node_idx.append(i)
                remain_node_idx = torch.tensor(remain_node_idx)
            hidden_states = self.fusion(hidden_states + self.dropout(attention_output[0]), gat_output)
            # hidden_states = (hidden_states + self.dropout(attention_output[0]) + gat_output) / 2
        else:
            if seq_idx2graph_idx is not None:
                for i in range(seq_idx2graph_idx.shape[0]):
                    remain_node_idx.append(i)
            remain_node_idx = torch.tensor(remain_node_idx)
            hidden_states = hidden_states + self.dropout(attention_output[0])
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        # if gat_edges is not None and gat_edges.shape[1] > 0 and gat_edges.max() >= hidden_states.shape[0]:
        #     print("")
        return outputs, seq_idx2graph_idx, gat_edges, remain_node_idx


class T5TreeBlock(GradientCheckpointingLayer):
    def __init__(self, config, has_relative_attention_bias=False, layer_idx: Optional[int] = None, fusion=None, gat_layer=None):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias,
                                 layer_idx=layer_idx, fusion=fusion, gat_layer=gat_layer)
        )
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config, layer_idx=layer_idx))

        self.layer.append(T5LayerFF(config))

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_decoder_position_bias=None,
            layer_head_mask=None,
            cross_attn_layer_head_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
            cache_position=None,
            node_to_sample_ptr=None,
            edge_to_sample_ptr=None,
            gat_edges=None,
            seq_idx2graph_idx=None
    ):
        hidden_states_new = None
        attention_outputs = None
        attention_mask_new = None
        position_bias_new = None
        layer_head_mask_new = None
        past_key_values_new = None

        totol = 0
        node_to_sample_ptr_new = [0]
        seq_idx2graph_idx_new = None
        gat_edges_new = None

        if node_to_sample_ptr is None:
            self_attention_outputs, _, _, _ = self.layer[0](
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                layer_head_mask=layer_head_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                cache_position=cache_position
            )
            hidden_states = self_attention_outputs[0]
            attention_outputs = self_attention_outputs[1:]
        else:
            # batch_size = len(node_to_sample_ptr) - 1
            # group_sizes = node_to_sample_ptr[1:] - node_to_sample_ptr[:-1]
            # group_indices = torch.repeat_interleave(
            #     torch.arange(batch_size, device=hidden_states.device),
            #     repeats=group_sizes,
            #     dim=0
            # )
            #
            # gat_edges_all = []
            # for i in range(batch_size):
            #     edges = gat_edges[i]
            #     group_id = torch.full((edges.shape[0], 1), i, device=edges.device)
            #     gat_edges_all.append(torch.cat([edges, group_id], dim=1))
            # gat_edges_all = torch.cat(gat_edges_all, dim=0)
            for i in range(len(node_to_sample_ptr) - 1):
                start_idx = node_to_sample_ptr[i]
                end_idx = node_to_sample_ptr[i + 1]

                batch_hidden_states = hidden_states[start_idx: end_idx, :, :]

                batch_attention_mask = attention_mask[start_idx: end_idx, :, :, :] if \
                    attention_mask is not None else None

                batch_position_bias = position_bias[start_idx: end_idx, :, :, :] if \
                    position_bias is not None else None

                batch_layer_head_mask = layer_head_mask[start_idx: end_idx, :, :] if \
                    layer_head_mask is not None else None

                batch_past_key_values = past_key_values[start_idx: end_idx, :, :] if \
                    past_key_values is not None else None

                batch_gat_edges = gat_edges[i]

                batch_seq_idx2graph_idx = seq_idx2graph_idx[start_idx: end_idx]

                # 获取组内保留索引
                self_attention_outputs, seq_idx2graph_idx_sub, gat_edge, remain_node_idx = self.layer[0](
                    batch_hidden_states,
                    attention_mask=batch_attention_mask,
                    position_bias=batch_position_bias,
                    layer_head_mask=batch_layer_head_mask,
                    past_key_values=batch_past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                    gat_edges=batch_gat_edges,
                    seq_idx2graph_idx=batch_seq_idx2graph_idx
                )
                idx = 0
                if attention_mask is None:
                    attention_mask_new = None
                else:
                    if attention_mask_new is None:
                        attention_mask_new = []
                    attention_mask_new.append(batch_attention_mask[remain_node_idx])

                if position_bias is not None:
                    if position_bias_new is None:
                        position_bias_new = []
                    position_bias_new.append(batch_position_bias[remain_node_idx])
                else:
                    position_bias_new = None

                if layer_head_mask is not None:
                    if layer_head_mask_new is None:
                        layer_head_mask_new = []
                    layer_head_mask_new.append(batch_layer_head_mask[remain_node_idx])
                else:
                    layer_head_mask_new = None

                if past_key_values is not None:
                    if past_key_values_new is None:
                        past_key_values_new = []
                    past_key_values_new.append(batch_past_key_values[remain_node_idx])
                else:
                    past_key_values_new = None

                if hidden_states_new is None:
                    hidden_states_new = []
                hidden_states_new.append(self_attention_outputs[0])
                totol += self_attention_outputs[0].shape[0]
                node_to_sample_ptr_new.append(totol)

                if seq_idx2graph_idx_new is None:
                    seq_idx2graph_idx_new = []

                seq_idx2graph_idx_new.append(seq_idx2graph_idx_sub)

                if gat_edges_new is None:
                    gat_edges_new = []
                gat_edges_new.append(gat_edge)

                if attention_outputs is None:
                    attention_outputs = []
                attention_outputs.append(self_attention_outputs[1])

            # 批量处理
            if attention_mask is not None:
                attention_mask = torch.cat(attention_mask_new, dim=0)

            if position_bias is not None:
                position_bias = torch.cat(position_bias_new, dim=0)

            if layer_head_mask is not None:
                layer_head_mask = torch.cat(layer_head_mask_new, dim=0)

            if past_key_values is not None:
                past_key_values = torch.cat(past_key_values_new, dim=0)

            if seq_idx2graph_idx is not None:
                seq_idx2graph_idx = torch.cat(seq_idx2graph_idx_new, dim=0)

            gat_edges = gat_edges_new
            node_to_sample_ptr = node_to_sample_ptr_new
            hidden_states = torch.cat(hidden_states_new, dim=0)
            attention_outputs = (
                torch.cat(attention_outputs, dim=0),)  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_values=past_key_values,
                query_length=cache_position[-1] + 1,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[1:]

        # Apply Feed Forward layer
        if self.layer[0].SelfAttention.layer_idx < 6 and not self.is_decoder:
            hidden_states_new = hidden_states.clone()
            for i in range(len(node_to_sample_ptr) - 1):
                start_idx = node_to_sample_ptr[i]
                end_idx = node_to_sample_ptr[i + 1]
                if start_idx == end_idx:
                    break
                hidden_states_new[start_idx: end_idx, :, :] = self.layer[-1](hidden_states[start_idx: end_idx, :, :])
            hidden_states = hidden_states_new.clone()
        else:
            hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return (
                       outputs + attention_outputs
               ), node_to_sample_ptr, seq_idx2graph_idx, gat_edges, \
               attention_mask, position_bias, layer_head_mask, past_key_values  # hidden-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5WithTreeStack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.block = nn.ModuleList([
                T5TreeBlock(config, has_relative_attention_bias=bool(i == 0), layer_idx=i)
                for i in range(config.num_layers)
            ])
        else:
            # only encoder has gat
            # TODO add 3 groups of GAT
            self.gat_first = CodeAwareGAT(
                config=config,
                hidden_size=config.d_model,
                heads=4
            )  # GAT for SelfAttention Layer 1 and 2
            self.gat_second = CodeAwareGAT(
                config=config,
                hidden_size=config.d_model,
                heads=4
            )  # GAT for SelfAttention Layer 3 and 4
            self.gat_third = CodeAwareGAT(
                config=config,
                hidden_size=config.d_model,
                heads=4
            )  # GAT for SelfAttention Layer 5 and 6
            self.fusion = ConcatMLPFusion(d_target=config.d_model)
            self.block = nn.ModuleList()

            # group0
            self.block.append(T5TreeBlock(config, has_relative_attention_bias=True,
                                          layer_idx=0, fusion=self.fusion, gat_layer=self.gat_first))
            self.block.append(T5TreeBlock(config, has_relative_attention_bias=False,
                                          layer_idx=1, fusion=self.fusion, gat_layer=self.gat_first))
            # group1
            self.block.append(T5TreeBlock(config, has_relative_attention_bias=False,
                                          layer_idx=2, fusion=self.fusion, gat_layer=self.gat_second))
            self.block.append(T5TreeBlock(config, has_relative_attention_bias=False,
                                          layer_idx=3, fusion=self.fusion, gat_layer=self.gat_second))
            # group2
            self.block.append(T5TreeBlock(config, has_relative_attention_bias=False,
                                          layer_idx=4, fusion=self.fusion, gat_layer=self.gat_third))
            self.block.append(T5TreeBlock(config, has_relative_attention_bias=False,
                                          layer_idx=5, fusion=self.fusion, gat_layer=self.gat_third))

            for i in range(6, config.num_layers):
                self.block.append(T5TreeBlock(config, has_relative_attention_bias=bool(i == 0), layer_idx=i))

        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def init_params(self):
        self.gat_first.gat_conv1.reset_parameters()
        self.gat_first.gat_conv2.reset_parameters()

        self.gat_second.gat_conv1.reset_parameters()
        self.gat_second.gat_conv2.reset_parameters()

        self.gat_third.gat_conv1.reset_parameters()
        self.gat_third.gat_conv2.reset_parameters()
        self.fusion.init_params()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5Stack.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            cache_position=None,
            node_to_sample_ptr=None,
            edge_to_sample_ptr=None,
            gat_edges=None,
            seq_idx2graph_idx=None
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            # # split inputs
            # for i in range(input_shape[0]):
            #     input_ids[i] = input_ids[i].view(-1, input_shape[-1])
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        if self.is_decoder:
            if use_cache and past_key_values is None:
                if self.config.is_encoder_decoder:
                    past_key_values = EncoderDecoderCache(
                        DynamicCache(config=self.config), DynamicCache(config=self.config)
                    )
                else:
                    past_key_values = DynamicCache(config=self.config)
        elif not self.is_decoder:
            # do not pass cache object down the line for encoder stack
            # it messes indexing later in decoder-stack because cache object is modified in-place
            past_key_values = None

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )

        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        if self.config.is_decoder:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values.self_attention_cache
                if isinstance(past_key_values, EncoderDecoderCache)
                else past_key_values,
                output_attentions,
            )
        elif attention_mask is not None:
            causal_mask = attention_mask[:, None, None, :]
            causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
            causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
        else:
            causal_mask = None

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, layer_module in enumerate(self.block):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if causal_mask is not None:
                    causal_mask = causal_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs, node_to_sample_ptr, seq_idx2graph_idx, gat_edges, \
            causal_mask, position_bias, layer_head_mask, past_key_values = layer_module(
                hidden_states,
                causal_mask,
                position_bias,
                encoder_hidden_states,
                encoder_extended_attention_mask,
                encoder_decoder_position_bias,  # as a positional argument for gradient checkpointing
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=return_dict,
                cache_position=cache_position,
                node_to_sample_ptr=node_to_sample_ptr,
                edge_to_sample_ptr=edge_to_sample_ptr,
                gat_edges=gat_edges,
                seq_idx2graph_idx=seq_idx2graph_idx
            )
            # TODO diffrent function for diffrent layer
            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1]

            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[3 if output_attentions else 2]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        tmp = []
        if node_to_sample_ptr is not None:
            for i in range(len(node_to_sample_ptr) - 1):
                start_idx = node_to_sample_ptr[i]
                end_idx = node_to_sample_ptr[i + 1]
                batch__hidden = hidden_states[start_idx: end_idx, :, :]
                tmp.append(batch__hidden[0, :, :].unsqueeze(0))
            hidden_states = torch.cat(tmp, dim=0).clone()
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    # Copied from transformers.models.gptj.modeling_gptj.GPTJModel._update_causal_mask
    def _update_causal_mask(
            self,
            attention_mask: Union[torch.Tensor, "BlockMask"],
            input_tensor: torch.Tensor,
            cache_position: torch.Tensor,
            past_key_values: Cache,
            output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                    attention_mask,
                    inputs_embeds=input_tensor,
                    past_key_values_length=past_seen_tokens,
                    is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
                self.config._attn_implementation == "sdpa"
                and attention_mask is not None
                and attention_mask.device.type in ["cuda", "xpu", "npu"]
                and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.gptj.modeling_gptj.GPTJModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask: torch.Tensor,
            sequence_length: int,
            target_length: int,
            dtype: torch.dtype,
            cache_position: torch.Tensor,
            batch_size: int,
            **kwargs,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class T6ForConditionalGeneration(T5PreTrainedModel, GenerationMixin):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.tie_encoder_decoder = False
        self.encoder = T5WithTreeStack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.tie_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5WithTreeStack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def init_params(self):
        self.encoder.init_params()

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def get_encoder(self):
        return self.encoder

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[tuple[tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            gat_edges: Optional[torch.LongTensor] = None,
            node_to_sample_ptr=None,
            edge_to_sample_ptr=None,
            seq_idx2graph_idx=None
    ) -> Union[tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        __HEAD_MASK_WARNING_MSG = """
        The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
        `decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
        If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
        num_heads)`.
        """
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask
        """
            input_ids -> shape(all_node_size, seq_length)
        """
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # 这里返回的形状应该是 (batch_size, seq_length, hidden_size)
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                node_to_sample_ptr=node_to_sample_ptr,
                edge_to_sample_ptr=edge_to_sample_ptr,
                gat_edges=gat_edges,
                seq_idx2graph_idx=seq_idx2graph_idx
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        attention_mask_length = 0
        if attention_mask is not None:
            attention_mask_length = len(attention_mask.shape)
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)
        # Decode
        encoder_attention_mask = attention_mask
        if attention_mask is not None and hidden_states is not None:
            if attention_mask.shape[0] > hidden_states.size(0):
                encoder_attention_mask = attention_mask[:hidden_states.size(0), :]
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        # TODO change the loss function
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)


class CodeAwareGAT(nn.Module):
    def __init__(self, config, hidden_size, top_k=None, heads=4):
        super().__init__()
        # 定义两层GATConv
        self.hidden_size = hidden_size
        # self.conv = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.gat_conv1 = CodeAwareGATConv(in_channels=hidden_size, out_channels=hidden_size,
                                          heads=heads, concat=True, add_self_loops=True)
        self.gat_conv2 = CodeAwareGATConv(in_channels=hidden_size * heads, out_channels=hidden_size,
                                          heads=1, concat=False, add_self_loops=True)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.top_k = top_k

    def forward(self, x, edge_index):
        """`
        x: 节点特征矩阵, shape: (num_nodes, seq_length, hidden_size)
        edge_index: 边索引, shape: (2, num_edges)
        """
        # --- GAT标准前向传播 ---
        # 第一层
        seq_length = x.shape[1]
        hidden_transposed = x.transpose(1, 2)

        node_feat = hidden_transposed[:, :, 0]

        gat_output = self.gat_conv1(node_feat, edge_index)
        gat_output = F.relu(gat_output)
        gat_output = F.dropout(gat_output, p=0.1, training=self.training)

        gat_output, attn = self.gat_conv2(gat_output, edge_index, return_attention_weights=True)

        gat_output_3d = gat_output.unsqueeze(1).repeat(1, seq_length, 1)

        gat_output_3d = self.layer_norm(x + gat_output_3d)

        return gat_output_3d, attn


__all__ = [
    "T6ForConditionalGeneration"
]
