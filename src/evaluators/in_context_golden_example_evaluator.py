import time
import torch
from torch import Tensor
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from .in_context_evaluator import RandomInContextEvaluator, RESPONSE_MODEL, HIDDEN_STATES_MODEL
from enum import Enum


class InformationInContextGoldenExampleEvaluator(RandomInContextEvaluator):
    def sample_and_evaluate_few_shot_quality(self, xq_embeddings: Dict[str, Tensor]):
        golden_examples_sample_times = self.config.get("golden_examples_sample_times", 10)
        best_lambda_1 = float('inf')
        best_results = {}

        last_layer_name = f"layer_{self.model.layer_num}"
        assert last_layer_name in xq_embeddings.keys(), f"last_layer_name: {last_layer_name} not in xq_embeddings.keys()"

        for _ in range(golden_examples_sample_times):

            (
                all_xi_all_y_embeddings, 
                all_xi_yi_embeddings, 
                few_shot_examples
            ) = self.sample_few_shot_examples()

            # 计算lambda_1
            Xi_matrix, Xi_pinv = self.solve_Xi_matrix(all_xi_all_y_embeddings[last_layer_name])
            lambda_1 = self.solve_lambda_1_Xq_Xi_dagger(Xi_pinv, xq_embeddings[last_layer_name]).cpu()

            # 更新 results
            if lambda_1 < best_lambda_1:
                best_lambda_1 = lambda_1
                best_results = {
                    "lambda_1": lambda_1, 
                    "few_shot_examples": few_shot_examples, 
                    "all_xi_all_y_embeddings": all_xi_all_y_embeddings,
                    "all_xi_yi_embeddings": all_xi_yi_embeddings,
                    "Xi_matrix": Xi_matrix,
                    "Xi_pinv": Xi_pinv
                }
        
        return best_results

    def evaluate_single_example(self, test_item: Dict[str, Any], extraction_layers: List[str]) -> Dict[str, Any]:
        """评估单个测试样例"""
        # 获得 \xi(x_Q)
        xq_embeddings, _ = self.sample_embeddings(test_item)

        # 采样并评估few-shot quality
        now_time = time.time()

        last_layer_name = f"layer_{self.model.layer_num}"
        best_results = self.sample_and_evaluate_few_shot_quality(xq_embeddings)
        
        Xi_pinv = best_results["Xi_pinv"].to(self.model.device)
        all_xi_yi_embeddings = best_results["all_xi_yi_embeddings"][last_layer_name].to(self.model.device)
        few_shot_examples = best_results["few_shot_examples"]
        lambda_1 = {
            last_layer_name: best_results["lambda_1"]
        }
        Xi_matrix = best_results["Xi_matrix"].to(self.model.device)
        rank = {
            last_layer_name: torch.linalg.matrix_rank(Xi_matrix)
        }

        # 计算每个层的 \bar{\xi}(x_i, y_i)
        mean_xi_yi_embeddings = torch.mean(all_xi_yi_embeddings, dim=0)

        # 计算 \alpha
        alpha = {
            last_layer_name: self.solve_alpha(Xi_pinv, mean_xi_yi_embeddings).to(self.model.device)
        }
        xq_embeddings_cuda = {k: v.to(self.model.device) for k, v in xq_embeddings.items()}

        argmax_hat_P_cuda = self.compute_predictions(xq_embeddings_cuda, alpha)
        argmax_hat_P = {k: v.cpu() for k, v in argmax_hat_P_cuda.items()}
        solve_time = time.time() - now_time
        
        return {
            'predictions': argmax_hat_P,
            'few_shot_examples': few_shot_examples,
            'solve_time': solve_time,
            'lambda_1': lambda_1,
            'rank': rank
        }
