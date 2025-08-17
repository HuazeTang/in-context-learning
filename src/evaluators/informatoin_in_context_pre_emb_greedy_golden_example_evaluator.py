import time
import torch
from torch import Tensor
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from .information_in_context_random_pre_emb import RandomInforInContextEvaluatorPreEmb
from .information_in_context_golden_example_evaluator import InformationInContextGoldenExampleEvaluator


PARALLEL_BATCH_SIZE = 32

class InformationInContextPreEmbGreedyGoldenExampleEvaluator(RandomInforInContextEvaluatorPreEmb, InformationInContextGoldenExampleEvaluator):
    def get_all_dev_samples_Xi(self, extraction_layers: List[str], pool_method: str) -> Tuple[Tensor, List[Dict[str, Any]]]:
        r"""获取所有dev样本的 \Xi 矩阵及其逆矩阵"""
        # self.all_Xi_matrix = defaultdict(list)
        self.all_Xi_pinv = defaultdict(list)
        for item in tqdm(self.dataset.dev_data, desc="Building dev data embeddings"):  
            x_embeddings, _ = self.sample_embeddings(item, extraction_layers, pool_method)
            x_embeddings = x_embeddings[0]
            for layer_name in x_embeddings.keys():
                _, Xi_pinv = self.solve_Xi_matrix(x_embeddings[layer_name].unsqueeze(0))
                self.all_Xi_pinv[layer_name].append(Xi_pinv)
    
    @torch.inference_mode()
    def sample_and_evaluate_few_shot_quality(self, xq_embeddings: Dict[str, Tensor], extraction_layers: List[str], pool_method: str):
        best_results = {}
        num_samples = self.config.get("num_shots", 5)

        # 并行化评估sample质量
        last_layer_name = f"layer_{self.model.layer_num}"
        assert last_layer_name in xq_embeddings.keys(), \
            f"last_layer_name: {last_layer_name} not in xq_embeddings.keys()"

        lambda_1_all = torch.zeros((len(self.dataset.dev_data),), device=xq_embeddings[last_layer_name].device)
        for i in range(0, len(self.dataset.dev_data), PARALLEL_BATCH_SIZE):
            batch_samples_invs = torch.stack(
                self.all_Xi_pinv[last_layer_name][i:min(i+PARALLEL_BATCH_SIZE, len(self.dataset.dev_data))],
                dim=0
            )
            batch_data_len = batch_samples_invs.shape[0]
            batch_lambda_1 = self.parallel_solve_lambda_1_Xq_Xi_dagger(batch_samples_invs, xq_embeddings[last_layer_name])
            lambda_1_all[i:i+batch_data_len] = batch_lambda_1

        # filter out those lambda_1 < 1
        mask = lambda_1_all < 1
        lambda_1_all[mask] = 10.0
        
        # find the optimal (min) num_shots lambda_1
        top_k_pos = torch.topk(lambda_1_all, k=num_samples, largest=False).indices.cpu().tolist()
        all_few_samples = [self.dataset.dev_data[i] for i in top_k_pos]

        all_xi_all_y_embeddings = self.get_all_xi_all_y_embeddings(all_few_samples, extraction_layers)
        all_xi_yi_embeddings = self.get_all_xi_yi_embeddings(all_few_samples, extraction_layers)

        Xi_matrix, Xi_pinv = self.solve_Xi_matrix(all_xi_all_y_embeddings[last_layer_name])

        lambda_1 = self.solve_lambda_1_Xq_Xi_dagger(Xi_pinv, xq_embeddings[last_layer_name].to(self.model.device))

        # 取得最优结果
        best_results = {
            "lambda_1": lambda_1, 
            "few_shot_examples": all_few_samples,
            "all_xi_all_y_embeddings": all_xi_all_y_embeddings,
            "all_xi_yi_embeddings": all_xi_yi_embeddings,
            "Xi_matrix": Xi_matrix,
            "Xi_pinv": Xi_pinv
        }
        
        return best_results
    
    def evaluate(self, output_dir: str = None) -> Tuple[float, List[Dict[str, Any]]]:
        last_layer_name = f"layer_{self.model.layer_num}"
        extraction_layers = [last_layer_name]
        pool_method = self.config.get('pool_method', None)
        self.get_all_dev_samples_Xi(extraction_layers, pool_method)
        return InformationInContextGoldenExampleEvaluator.evaluate(self, output_dir)
