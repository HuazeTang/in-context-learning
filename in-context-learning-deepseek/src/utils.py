import re
import numpy as np
import pickle
import torch
from datasets import load_dataset
import json
import os
from pathlib import Path
from typing import Optional
from omegaconf import DictConfig, OmegaConf

def extract_answer(text):
    """从模型输出中提取答案字母 (A, B, C, D)"""
    match = re.search(r'\b([A-D])\b', text)
    return match.group(1) if match else None

def load_mmlu_dataset(subject, split='test', max_samples: Optional[int]=None, dataset_path: Optional[str]=None):
    """加载MMLU数据集"""
    if dataset_path:
        dataset = load_dataset(dataset_path, subject)
    else:
        dataset = load_dataset("cais/mmlu", subject)
    data = dataset[split]
    if max_samples and max_samples > 0:
        data = data.select(range(min(max_samples, len(data))))
    return data

def build_few_shot_prompt(examples, num_shots=5):
    """构建few-shot提示"""
    prompt = ""
    for i in range(min(num_shots, len(examples))):
        ex = examples[i]
        question = ex["question"]
        choices = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(ex["choices"])])
        answer = chr(65 + ex["answer"])
        prompt += f"Question: {question}\n{choices}\nAnswer: {answer}\n\n"
    return prompt.strip()

def build_full_prompt(question, choices, few_shot_prompt):
    """构建完整提示"""
    choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    return (
        f"{few_shot_prompt}\n\n"
        f"Question: {question}\n{choices_text}\nAnswer:"
    )

def _convert_dictconfig(obj):
    """转换DictConfig对象为普通字典"""
    if isinstance(obj, DictConfig):
        return OmegaConf.to_container(obj, resolve=True)
    elif isinstance(obj, dict):
        return {k: _convert_dictconfig(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_dictconfig(item) for item in obj]
    else:
        return obj

def _get_hidden_states_shape_info(hidden_states):
    """提取hidden states的shape信息"""
    if hidden_states is None:
        return None
    
    shape_info = []
    for step_idx, step_layers in enumerate(hidden_states):
        step_info = {}
        for layer_name, layer_data in step_layers.items():
            if isinstance(layer_data, np.ndarray) or isinstance(layer_data, torch.Tensor):
                step_info[layer_name] = {
                    "shape": list(layer_data.shape),
                    "dtype": str(layer_data.dtype)
                }
            else:
                step_info[layer_name] = {
                    "shape": "unknown",
                    "dtype": str(type(layer_data))
                }
        shape_info.append(step_info)
    return shape_info

def save_results(results, output_dir):
    """保存评估结果到文件
    
    保存两个文件：
    1. results.json - 包含shape信息的轻量级结果
    2. full_results.pkl - 包含完整hidden states的完整结果
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / "results.json"
    
    # 处理DictConfig对象    
    serializable_results = _convert_dictconfig(results)

    # 保存完整结果到pickle
    # pickle_path = Path(output_dir) / "full_results.pkl"
    # with open(pickle_path, 'wb') as f:
    #     pickle.dump(serializable_results, f)
    # print(f"Full results (with hidden states) saved to {pickle_path}")
    
    # 准备JSON结果（hidden states只保存shape信息）
    json_results = serializable_results.copy()

    if 'results' in json_results:
        json_results['results'] = []
        for result in serializable_results['results']:
            json_result = result.copy()
            if result.get('hidden_states') is not None:
                json_result['hidden_states'] = _get_hidden_states_shape_info(result['hidden_states'])
            json_results['results'].append(json_result)
    
    # 添加说明
    json_results['note'] = "Hidden states shapes are shown here. Full hidden states data is saved in full_results.pkl"
    
    # 保存JSON结果
    json_path = Path(output_dir) / "results.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {output_path}")
    return output_path
