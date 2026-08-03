# coding=utf-8
"""
MIND (DyKo3Layer): 基于认知动机的层次对齐框架。
移除了原有的 CLCR 解耦模块，严格按照 MIND 论文的前向逻辑重构：
1. Coarse slide context 筛选 entity anchors
2. Grounding 到 low-mag visual evidence
3. Entity context 约束 high-mag cell evidence aggregation
4. Slide & Cell 分支联合预测
"""
from __future__ import absolute_import, division, print_function

import json
import os
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .DyKo import Adapter, encode_text
from .model_utils import *

try:
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
except Exception:
    create_model_from_pretrained = get_tokenizer = None


class TextEncoder(nn.Module):
    # 保持不变
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.text_encoder.transformer
        self.positional_embedding = clip_model.text_encoder.positional_embedding
        self.ln_final = clip_model.text_encoder.ln_final
        self.text_projection = clip_model.text_encoder.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding[:-1, :]
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        row_indices = torch.arange(x.shape[0], device=x.device)
        eot_indices = tokenized_prompts.argmax(dim=-1).to(x.device)
        x = x[row_indices, eot_indices] @ self.text_projection
        return x


class LearnedPromptLearner(nn.Module):
    # 保持不变
    def __init__(self, slide_classnames, slide_descriptions, entity_names, entity_descriptions, cell_class_names,
                 cell_descriptions, clip_model, n_ctx=16):
        super().__init__()
        device = next(clip_model.parameters()).device
        self.n_ctx = n_ctx
        ctx_dim = clip_model.text_encoder.text_projection.shape[1]

        prompt_prefix = " ".join(["X"] * n_ctx)

        slide_prompts = []
        for name, desc in zip(slide_classnames, slide_descriptions):
            slide_prompts.append(f"{prompt_prefix} {name.replace('_', ' ')} {desc}")
        self.n_slide = len(slide_classnames)

        entity_prompts = []
        if entity_names:
            for name, desc in zip(entity_names, entity_descriptions):
                entity_prompts.append(f"{prompt_prefix} {name.replace('_', ' ')} {desc}")
            self.n_entity = len(entity_names)
        else:
            self.n_entity = 0

        cell_prompts = []
        if cell_class_names:
            for name, desc in zip(cell_class_names, cell_descriptions):
                cell_prompts.append(f"{prompt_prefix} {name.replace('_', ' ')} {desc}")
            self.n_cell_classes = len(cell_class_names)
        else:
            self.n_cell_classes = 0

        self.slide_ctx = nn.Parameter(torch.empty(self.n_slide, n_ctx, ctx_dim))
        self.entity_ctx = nn.Parameter(torch.empty(self.n_entity, n_ctx, ctx_dim)) if self.n_entity > 0 else None
        self.cell_ctx = nn.Parameter(
            torch.empty(self.n_cell_classes, n_ctx, ctx_dim)) if self.n_cell_classes > 0 else None

        nn.init.normal_(self.slide_ctx, std=0.02)
        if self.entity_ctx is not None:
            nn.init.normal_(self.entity_ctx, std=0.02)
        if self.cell_ctx is not None:
            nn.init.normal_(self.cell_ctx, std=0.02)

        tokenizer = getattr(clip_model.text_encoder, 'tokenizer', None)
        if tokenizer is None and hasattr(clip_model, 'tokenizer'):
            tokenizer = clip_model.tokenizer

        if tokenizer is None:
            raise ValueError("No tokenizer found in CLIP model")

        all_prompts = slide_prompts + entity_prompts + cell_prompts
        tokenized_prompts = tokenizer(all_prompts).to(device)

        with torch.no_grad():
            _, embedding = clip_model.encode_text(tokenized_prompts)

        slide_end = self.n_slide
        entity_end = slide_end + self.n_entity
        cell_end = entity_end + self.n_cell_classes

        self.register_buffer("slide_token_prefix", embedding[:slide_end, :1, :])
        self.register_buffer("slide_token_suffix", embedding[:slide_end, 1 + n_ctx:, :])

        if self.n_entity > 0:
            self.register_buffer("entity_token_prefix", embedding[slide_end:entity_end, :1, :])
            self.register_buffer("entity_token_suffix", embedding[slide_end:entity_end, 1 + n_ctx:, :])

        if self.n_cell_classes > 0:
            self.register_buffer("cell_token_prefix", embedding[entity_end:cell_end, :1, :])
            self.register_buffer("cell_token_suffix", embedding[entity_end:cell_end, 1 + n_ctx:, :])

        self.tokenized_prompts = tokenized_prompts

    def forward(self, text_encoder):
        slide_prefix = self.slide_token_prefix
        slide_suffix = self.slide_token_suffix
        slide_prompts = torch.cat([slide_prefix, self.slide_ctx, slide_suffix], dim=1)
        slide_feat = text_encoder(slide_prompts, self.tokenized_prompts[: self.n_slide])
        results = {'slide': slide_feat}

        if self.n_entity > 0:
            entity_prefix = self.entity_token_prefix
            entity_suffix = self.entity_token_suffix
            entity_prompts = torch.cat([entity_prefix, self.entity_ctx, entity_suffix], dim=1)
            entity_feat = text_encoder(entity_prompts,
                                       self.tokenized_prompts[self.n_slide: self.n_slide + self.n_entity])
            results['entity'] = entity_feat

        if self.n_cell_classes > 0:
            cell_prefix = self.cell_token_prefix
            cell_suffix = self.cell_token_suffix
            cell_prompts = torch.cat([cell_prefix, self.cell_ctx, cell_suffix], dim=1)
            start_idx = self.n_slide + self.n_entity
            cell_feat = text_encoder(cell_prompts, self.tokenized_prompts[start_idx: start_idx + self.n_cell_classes])
            results['cell'] = cell_feat

        return results

    def set_trainable_ctx_only(self):
        for p in self.parameters():
            p.requires_grad = False
        self.slide_ctx.requires_grad_(True)
        if self.entity_ctx is not None:
            self.entity_ctx.requires_grad_(True)
        if self.cell_ctx is not None:
            self.cell_ctx.requires_grad_(True)


def _encode_strings_with_clip(clip_model, tokenizer, strings, device, max_length=77):
    # 保持不变
    if tokenizer is None:
        raise ValueError("tokenizer is None, cannot encode 3-layer prompts.")
    try:
        if hasattr(tokenizer, '__call__'):
            out = tokenizer(strings)
            if isinstance(out, dict):
                input_ids = out['input_ids'].to(device)
            else:
                input_ids = (out[0] if isinstance(out, (list, tuple)) else out).to(device)
        else:
            input_ids = tokenizer(strings).to(device)
    except Exception as e:
        input_ids_list = []
        for s in strings:
            try:
                if hasattr(tokenizer, '__call__'):
                    out = tokenizer(s)
                    if isinstance(out, dict):
                        ids = out['input_ids']
                    else:
                        ids = out[0] if isinstance(out, (list, tuple)) else out
                else:
                    ids = tokenizer(s)
                input_ids_list.append(ids)
            except Exception as inner_e:
                input_ids_list.append([0] * max_length)

        if input_ids_list:
            input_ids = torch.tensor(input_ids_list, device=device)
        else:
            input_ids = torch.zeros(len(strings), max_length, dtype=torch.long, device=device)

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    if input_ids.shape[1] < max_length:
        padding_length = max_length - input_ids.shape[1]
        padding_tensor = torch.zeros((input_ids.shape[0], padding_length), dtype=input_ids.dtype, device=device)
        input_ids = torch.cat([input_ids, padding_tensor], dim=1)
    elif input_ids.shape[1] > max_length:
        input_ids = input_ids[:, :max_length]

    with torch.no_grad():
        text_latent, _ = clip_model.encode_text(input_ids)
    return text_latent.float()


class Prompt3LayerModule(nn.Module):
    # 保持不变，负责管理 3 层 Text Anchors
    def __init__(self, prompt_3layer_path, clip_model, tokenizer, feature_dim, device, use_learned_prompts=False,
                 learned_prompt_learner=None, slide_class_order=None):
        super().__init__()
        self.use_learned_prompts = use_learned_prompts
        self.feature_dim = feature_dim
        self.clip_model = clip_model
        self.tokenizer = tokenizer
        self.device = device

        if use_learned_prompts:
            if learned_prompt_learner is None:
                raise ValueError("learned_prompt_learner must be provided when use_learned_prompts=True")
            self.learned_prompt_learner = learned_prompt_learner
            self.text_encoder = TextEncoder(clip_model)
            self.slide_class_names, self.entity_names, self.cell_class_names = [], [], []
            self.cell_focus_order, self.cell_focus_to_slice, self.entity_related_focus = [], {}, []
            self.n_slide, self.n_entity = 0, 0
        else:
            with open(os.path.expanduser(prompt_3layer_path), 'r', encoding='utf-8') as f:
                self.prompt_data = json.load(f)

            slide = self.prompt_data.get('slide_level', {})
            entity = self.prompt_data.get('entity_level', {})
            cell = self.prompt_data.get('cell_level', {})

            self.slide_class_names = list(slide.keys())
            if slide_class_order is not None:
                self.slide_class_names = list(slide_class_order)
            slide_texts = [slide[name].get('description', '') for name in self.slide_class_names]

            entity_list = entity.get('entities', [])
            entity_texts = [e.get('description', '') for e in entity_list]
            self.entity_names = [e.get('name', '') for e in entity_list]

            cell_focus_order = list(cell.keys())
            if 'description' in cell_focus_order:
                cell_focus_order.remove('description')
            self.cell_focus_order = [k for k in cell_focus_order if isinstance(cell.get(k), dict)]

            cell_texts = []
            self.cell_focus_to_slice = {}
            self.cell_class_names = None

            for focus in self.cell_focus_order:
                d = cell[focus]
                if isinstance(d, dict):
                    class_names = [k for k in d.keys() if k != 'description']
                    if self.cell_class_names is None:
                        self.cell_class_names = class_names
                    start_idx = len(cell_texts)
                    for class_name in self.cell_class_names:
                        if class_name in d:
                            cell_texts.append(d[class_name])
                    self.cell_focus_to_slice[focus] = (start_idx, len(cell_texts))

            slide_feat = _encode_strings_with_clip(clip_model, tokenizer, slide_texts, device)
            entity_feat = _encode_strings_with_clip(clip_model, tokenizer, entity_texts,
                                                    device) if entity_texts else torch.zeros(0, feature_dim,
                                                                                             device=device)
            cell_feat = _encode_strings_with_clip(clip_model, tokenizer, cell_texts,
                                                  device) if cell_texts else torch.zeros(0, feature_dim, device=device)

            self.register_buffer('slide_text_feat', slide_feat)
            self.register_buffer('entity_text_feat', entity_feat)
            self.register_buffer('cell_text_feat', cell_feat)
            self.n_slide = slide_feat.shape[0]
            self.n_entity = entity_feat.shape[0]

    def get_prompt_feats(self):
        if self.use_learned_prompts:
            prompts = self.learned_prompt_learner(self.text_encoder)
            return {'slide': prompts['slide'].float(),
                    'entity': prompts.get('entity', torch.zeros(0, self.feature_dim, device=self.device)).float(),
                    'cell': prompts.get('cell', torch.zeros(0, self.feature_dim, device=self.device)).float()}
        return {'slide': self.slide_text_feat, 'entity': self.entity_text_feat, 'cell': self.cell_text_feat}

    def get_cell_class_names(self):
        return getattr(self, 'cell_class_names', []) if self.use_learned_prompts else (self.cell_class_names or [])

    def get_slide_class_names(self):
        return self.slide_class_names


class MIND(nn.Module):
    """
    遵循 MIND 论文的架构
    """

    def __init__(
            self,
            input_size=768,
            prompt_3layer_path='./prompt/camelyon16_pathology_prompts_3layer_en.json',
            num_heads=8,
            n_classes=None,
            encoder_model='CONCH',
            entity_topk=3,
            use_learned_prompts=False,
            prompt_n_ctx=16,
            slide_class_order=None,
            # MIND 专属超参数
            tau_s=0.07,
            tau_c=0.07,
            beta=1.0,
            lambda_s=0.5,
            lambda_c=0.5,
    ):
        super().__init__()
        self.entity_topk = entity_topk
        self.use_learned_prompts = use_learned_prompts

        # 论文参数设置
        self.tau_s = nn.Parameter(torch.tensor(tau_s))
        self.tau_c = nn.Parameter(torch.tensor(tau_c))  # 可学习的 temperature parameter
        self.beta = nn.Parameter(torch.tensor(beta))  # entity-level conditioning 的控制强度
        self.lambda_s = lambda_s
        self.lambda_c = lambda_c

        if encoder_model == 'TITAN':
            clip_model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
            clip_model.encode_text = types.MethodType(encode_text, clip_model)
        elif encoder_model == 'CONCH' and create_model_from_pretrained is not None and get_tokenizer is not None:
            CONCH_CKPT_PATH = '/home/junjianli/Workflow-TCGA-Slides-for-MIL/FModel/CONCH/pytorch_model.bin'
            clip_model, _ = create_model_from_pretrained("conch_ViT-B-16", CONCH_CKPT_PATH)
            clip_model.text_encoder = clip_model.text
            clip_model.text_encoder.tokenizer = get_tokenizer()
        else:
            raise ValueError("Only supports 'TITAN' or 'CONCH'.")

        _ = clip_model.eval()
        self.device = next(clip_model.parameters()).device
        clip_model = clip_model.to(self.device)
        self.feature_dim = clip_model.text_encoder.text_projection.shape[1]

        tokenizer = getattr(clip_model.text_encoder, 'tokenizer', None) or getattr(clip_model, 'tokenizer', None)

        if self.use_learned_prompts:
            with open(os.path.expanduser(prompt_3layer_path), 'r', encoding='utf-8') as f:
                temp_prompt_data = json.load(f)

            slide_class_names = list(temp_prompt_data.get('slide_level', {}).keys())
            slide_descriptions = [temp_prompt_data['slide_level'][name].get('description', '') for name in
                                  slide_class_names]

            entity_list = temp_prompt_data.get('entity_level', {}).get('entities', [])
            entity_names = [e.get('name', '') for e in entity_list]
            entity_descriptions = [e.get('description', '') for e in entity_list]

            cell_data = temp_prompt_data.get('cell_level', {})
            cell_class_names, cell_descriptions = None, None
            cell_focus_order = list(cell_data.keys())
            if 'description' in cell_focus_order:
                cell_focus_order.remove('description')
            for focus in cell_focus_order:
                d = cell_data.get(focus)
                if isinstance(d, dict):
                    class_names = [k for k in d.keys() if k != 'description']
                    descriptions = [d[k] for k in class_names]
                    if cell_class_names is None:
                        cell_class_names = class_names
                        cell_descriptions = descriptions
                    break

            self.learned_prompt_learner = LearnedPromptLearner(
                slide_class_names, slide_descriptions, entity_names, entity_descriptions,
                cell_class_names or [], cell_descriptions or [], clip_model.float(), n_ctx=prompt_n_ctx
            )
            for p in clip_model.parameters():
                p.requires_grad = False
            self.learned_prompt_learner.set_trainable_ctx_only()

            self.prompt_3layer = Prompt3LayerModule(
                prompt_3layer_path, clip_model.float(), tokenizer, self.feature_dim, self.device,
                use_learned_prompts=True, learned_prompt_learner=self.learned_prompt_learner,
                slide_class_order=slide_class_order,
            )
            self.prompt_3layer.slide_class_names = slide_class_names
            self.prompt_3layer.entity_names = entity_names
            self.prompt_3layer.cell_class_names = cell_class_names or []
            self.prompt_3layer.n_slide = len(slide_class_names)
            self.prompt_3layer.n_entity = len(entity_names)
        else:
            self.prompt_3layer = Prompt3LayerModule(
                prompt_3layer_path, clip_model.float(), tokenizer, self.feature_dim, self.device,
                slide_class_order=slide_class_order,
            )
        self.n_entity = self.prompt_3layer.n_entity
        self.n_classes = n_classes if n_classes is not None else len(self.prompt_3layer.slide_class_names)

        # 映射 cell logits 到 slide class
        self.slide_class_names = self.prompt_3layer.get_slide_class_names()
        cell_class_names_for_mapping = self.prompt_3layer.get_cell_class_names()
        mapping_tensor = None
        if self.slide_class_names and cell_class_names_for_mapping:
            name_to_idx = {name: i for i, name in enumerate(self.slide_class_names)}
            lower_name_to_idx = {name.lower(): i for i, name in enumerate(self.slide_class_names)}
            mapping = []
            for cname in cell_class_names_for_mapping:
                idx = name_to_idx.get(cname, None)
                if idx is None:
                    idx = lower_name_to_idx.get(cname.lower(), -1)
                mapping.append(idx if idx is not None else -1)
            if any(i >= 0 for i in mapping):
                mapping_tensor = torch.tensor(mapping, dtype=torch.long)
        if mapping_tensor is None:
            mapping_tensor = torch.empty(0, dtype=torch.long)
        self.register_buffer("cell_to_slide_index", mapping_tensor)

        # Visual Embedders
        self.Path_Adapter = Adapter(c_in=input_size, reduction=4)
        self.entity_patch_proj_low = nn.Linear(input_size,
                                               self.feature_dim) if input_size != self.feature_dim else nn.Identity()
        self.entity_patch_proj_cell = nn.Linear(input_size,
                                                self.feature_dim) if input_size != self.feature_dim else nn.Identity()

        # Entity Grounding Cross Attention
        self.cross_attention_entity = nn.MultiheadAttention(embed_dim=self.feature_dim, num_heads=num_heads,
                                                            batch_first=True)
        self.entity_norm = nn.LayerNorm(self.feature_dim)

    def forward(self, **kwargs):
        data = kwargs['data']
        h_low = data[0]
        h_cell = data[1] if len(data) > 1 else h_low

        device = h_low.device

        # --- 0. Multi-Scale Bag Construction ---
        X_l = self.Path_Adapter(h_low.clone().float().squeeze(0))
        X_l = self.entity_patch_proj_low(X_l)  # [N_l, d]

        X_h = self.Path_Adapter(h_cell.clone().float().squeeze(0))
        X_h = self.entity_patch_proj_cell(X_h)  # [N_h, d]

        prompt_feats = self.prompt_3layer.get_prompt_feats()
        Z_s = prompt_feats['slide']  # [C, d]
        Z_e = prompt_feats['entity']  # [K_e, d]
        Z_c = prompt_feats['cell']  # [K_c, d]

        N_l, d = X_l.shape
        N_h, _ = X_h.shape
        K_e = Z_e.shape[0]

        # 如果没有配置实体，直接走最简单的全局池化预测
        if K_e == 0:
            v_s_coarse = X_l.mean(dim=0, keepdim=True)
            l_s = F.cosine_similarity(v_s_coarse, Z_s, dim=-1) / self.tau_s
            Y_prob = F.softmax(l_s, dim=1)
            Y_hat = torch.topk(Y_prob, 1, dim=1)[1]
            return {'logits': l_s, 'Y_prob': Y_prob, 'Y_hat': Y_hat}

        # --- 1. Slide-level: Coarse-to-fine aggregation process ---
        # Coarse global descriptor
        v_s_coarse = X_l.mean(dim=0, keepdim=True)  # [1, d]

        # 筛选语义实体 (Entity Filtering)
        sim_e_s = F.cosine_similarity(Z_e, v_s_coarse, dim=-1)  # [K_e]
        top_Re = min(self.entity_topk, K_e)
        _, top_e_idx = torch.topk(sim_e_s, k=top_Re)
        Z_e_tilde = Z_e[top_e_idx]  # [R_e, d]

        # --- 2. Entity-level: Grounding via Cross-Attention ---
        Q = Z_e_tilde.unsqueeze(0)  # [1, R_e, d]
        KV = X_l.unsqueeze(0)  # [1, N_l, d]

        # \widetilde{V}^{e} = CrossAttn(\widetilde{Z}^{e}W_Q, X^{l}W_K, X^{l}W_V)
        V_e_tilde, _ = self.cross_attention_entity(Q, KV, KV)
        V_e_tilde = self.entity_norm(V_e_tilde + Q).squeeze(0)  # [R_e, d]

        # \tilde{v}^{s} = MeanPool(\widetilde{V}^{e})
        v_s_tilde = V_e_tilde.mean(dim=0, keepdim=True)  # [1, d]

        # --- 3. Cell-level: Gated Aggregation ---
        # Entity-consistency score: g_m = \max \mathrm{sim}(x_m^h, \tilde{v}_r^e)
        sim_h_e = F.normalize(X_h, dim=-1) @ F.normalize(V_e_tilde, dim=-1).T  # [N_h, R_e]
        g_m, _ = sim_h_e.max(dim=-1, keepdim=True)  # [N_h, 1]

        # Gated matching score: s_{j,m} = \mathrm{sim}(x_m^h, z_j^c)/\tau_c + \beta g_m
        sim_h_c = F.normalize(X_h, dim=-1) @ F.normalize(Z_c, dim=-1).T  # [N_h, K_c]
        s_jm = sim_h_c / self.tau_c + self.beta * g_m  # [N_h, K_c]

        # Aggregation weights & Aggregated Cell Features
        alpha_jm = F.softmax(s_jm, dim=0)  # [N_h, K_c]
        V_c = alpha_jm.T @ X_h  # \sum \alpha_{j,m} x_m^h  => [K_c, d]

        # --- 4. Joint Prediction ---
        # 4.1 Slide branch logits
        l_s = (F.normalize(v_s_tilde, dim=-1) @ F.normalize(Z_s, dim=-1).T) / self.tau_s  # [1, C]

        # 4.2 Cell branch logits
        # \ell_c^{\mathrm{cell}} = 1/|\mathcal I(c)| \sum \mathrm{sim}(v_j^c, z_j^c)/\tau_c
        sim_c_c = F.cosine_similarity(V_c, Z_c, dim=-1) / self.tau_c  # [K_c]
        l_cell = torch.zeros(1, self.n_classes, device=device)
        class_counts = torch.zeros(1, self.n_classes, device=device)

        mapping = self.cell_to_slide_index.to(device)
        valid = (mapping >= 0) & (mapping < self.n_classes)
        if valid.any():
            l_cell.index_add_(1, mapping[valid], sim_c_c[valid].unsqueeze(0))
            class_counts.index_add_(1, mapping[valid], torch.ones_like(sim_c_c[valid]).unsqueeze(0))

        # 求平均
        l_cell = l_cell / class_counts.clamp(min=1e-5)

        # 4.3 Final fused logits
        logits = self.lambda_s * l_s + self.lambda_c * l_cell

        # 预测结果
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(Y_prob, 1, dim=1)[1]

        # 计算 Entity Grounding Regularizer L_{ent}
        # \mathcal L_{ent} = \frac{1}{R_e}\sum (1-\mathrm{sim}(\tilde v_r^e,\tilde z_r^e))
        L_ent = 1.0 - F.cosine_similarity(V_e_tilde, Z_e_tilde, dim=-1).mean()

        results = {
            'logits': logits,
            'logits_slide': l_s,  # 用于辅助监督 L_aux
            'logits_cell': l_cell,  # 用于辅助监督 L_aux
            'loss_ent': L_ent,  # 实体对齐正则 L_ent
            'Y_prob': Y_prob,
            'Y_hat': Y_hat
        }

        return results