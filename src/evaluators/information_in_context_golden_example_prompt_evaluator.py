import time
import torch
from torch import Tensor
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from .informatoin_in_context_golden_example_evaluator import InformationInContextGoldenExampleEvaluator, RESPONSE_MODEL, HIDDEN_STATES_MODEL
from enum import Enum


class InformationInContextGoldenExamplePromptEvaluator(InformationInContextGoldenExampleEvaluator):
    def build_few_shot_prompt(self, examples: List[Dict[str, Any]]) -> str:
        """构建few-shot提示"""
        prompt = ""
        for ex in examples:
            question = self.dataset.format_question(ex)
            answer = self.dataset.format_answer(ex)
            prompt += f"{question}\nAnswer: {answer}\n\n"
        return prompt.strip()

    def build_full_prompt(self, example: Dict[str, Any], few_shot_prompt: str) -> str:
        """构建完整提示"""
        return f"{few_shot_prompt}\n\n{self.dataset.format_question(example)}\nAnswer:"

    def evaluate_single_example(self, test_item: Dict[str, Any], extraction_layers: List[str], pool_method: str) -> Dict[str, Any]:
        """评估单个测试样例"""
        # 获得 \xi(x_Q)
        xq_embeddings, _ = self.sample_embeddings(test_item, extraction_layers, pool_method)
        xq_embeddings = xq_embeddings[0]

        # 采样并评估few-shot quality
        now_time = time.time()

        last_layer_name = f"layer_{self.model.layer_num}"
        best_results = self.sample_and_evaluate_few_shot_quality(xq_embeddings, extraction_layers, pool_method)
        
        few_shot_examples = best_results["few_shot_examples"]
        lambda_1 = {
            last_layer_name: best_results["lambda_1"]
        }
        Xi_matrix = best_results["Xi_matrix"].to(self.model.device)
        rank = {
            last_layer_name: torch.linalg.matrix_rank(Xi_matrix)
        }

        # 使用 prompt 推理结果
        ## 构建完整提示
        few_shot_prompt = self.build_few_shot_prompt(few_shot_examples)
        prompt = self.build_full_prompt(test_item, few_shot_prompt)

        # 准备消息
        system_prompt = self.config.get('system_prompt')
        messages = self.prepare_messages(prompt, system_prompt)
        
        # 生成答案
        result = self.generate_response(messages, extract_hidden_states=False)
        assert RESPONSE_MODEL in result, f"Response not found in result: {result.keys()}"
        response = result[RESPONSE_MODEL]

        # 提取预测
        pred_answer = self.dataset.extract_prediction(response)
        # 将ABCD映射为0123
        option_mapping = {'A': torch.tensor(0), 'B': torch.tensor(1), 'C': torch.tensor(2), 'D': torch.tensor(3)}
        if pred_answer in option_mapping:
            pred_answer = option_mapping[pred_answer]
        else:
            raise ValueError(f"Invalid prediction: {pred_answer}")

        solve_time = time.time() - now_time
        
        return {
            'predictions': {last_layer_name: pred_answer},
            'few_shot_examples': few_shot_examples,
            'solve_time': solve_time,
            'lambda_1': lambda_1,
            'rank': rank,
            'response': response
        }
