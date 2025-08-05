import time
import torch
from torch import Tensor
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from .information_in_context_evaluator import (
    RandomInforInContextEvaluator, RESPONSE_MODEL, HIDDEN_STATES_MODEL
)


class InformationInContextGoldenExampleEvaluator(RandomInforInContextEvaluator):
    def parallel_solve_Xi_matrix(self, all_xi_all_y_embeddings: Tensor) -> Tuple[Tensor, Tensor]:
        """并行化计算Xi矩阵"""
        assert len(all_xi_all_y_embeddings.shape) == 4, \
            f"Invalid shape for all_xi_all_y_layer_emb: {all_xi_all_y_embeddings.shape}, should be (batch, num_x, num_y, K)"
        
        batch_size, num_x, num_y, K = all_xi_all_y_embeddings.shape
        
        # Xi_matrix = torch.zeros((batch_size, K, K), device=all_xi_all_y_layer_emb.device, dtype=all_xi_all_y_layer_emb.dtype)
        reshaped_emb = all_xi_all_y_embeddings.reshape(batch_size, num_x * num_y, K).float()
        reshaped_emb_T = reshaped_emb.transpose(-2, -1) # shape: (batch_size, K, num_x * num_y)
        Xi_matrix = torch.bmm(reshaped_emb_T, reshaped_emb) / num_x # shape: (batch_size, K, K)
        # 使用 torch.linalg.pinv 计算逆矩阵
        # 示例：
        # >>> a = torch.rand((3,2,2))
        # >>> a
        # tensor([[[0.3420, 0.1157],
        #         [0.2696, 0.4860]],

        #         [[0.7587, 0.5012],
        #         [0.5113, 0.8513]],

        #         [[0.7514, 0.2995],
        #         [0.3140, 0.4726]]])
        # >>> print(torch.bmm(a, torch.linalg.pinv(a)))
        # tensor([[[ 1.0000e+00,  2.9802e-08],
        #         [ 1.7881e-07,  1.0000e+00]],

        #         [[ 1.0000e+00,  1.7881e-07],
        #         [-3.5763e-07,  1.0000e+00]],

        #         [[ 1.0000e+00, -1.7881e-07],
        #         [-5.9605e-08,  1.0000e+00]]])
        Xi_pinv = torch.linalg.pinv(Xi_matrix) # shape: (batch_size, K, K)
        
        return Xi_matrix, Xi_pinv
    
    def parallel_solve_lambda_1_Xq_Xi_dagger(self, Xi_pinv: Tensor, xq_embeddings: Tensor) -> Tensor:
        """并行化计算lambda_1"""
        assert len(Xi_pinv.shape) == 3, f"Invalid shape for Xi_pinv: {Xi_pinv.shape}, should be (batch_size, K, K)"
        assert len(xq_embeddings.shape) == 2, f"Invalid shape for xq_embeddings: {xq_embeddings.shape}, should be (num_y, K)"

        num_y, K = xq_embeddings.shape

        assert Xi_pinv.shape[1:] == (K, K), f"Invalid shape for Xi_pinv: {Xi_pinv.shape}, should be (f{K}, f{K})"

        Xq_matrix = xq_embeddings.float().T @ xq_embeddings.float() # shape: (K, K)
        Xq_Xi_dagger_matrix = Xq_matrix @ Xi_pinv # shape: (batch_size, K, K)

        # 求解 Xq_Xi_dagger_matrix 的最大特征值
        lambda_1 = torch.linalg.eigvalsh(Xq_Xi_dagger_matrix).max(dim=-1).values # shape: (batch_size,)

        return lambda_1
    
    def sample_and_evaluate_few_shot_quality(self, xq_embeddings: Dict[str, Tensor], extraction_layers: List[str]):
        golden_examples_sample_times = self.config.get("golden_examples_sample_times", 10)
        best_results = {}

        # 一次性采样所有的 samples
        all_few_samples = []
        # t = time.time()
        for _ in range(golden_examples_sample_times):
            (
                all_xi_all_y_embeddings, 
                all_xi_yi_embeddings, 
                few_shot_examples
            ) = self.sample_few_shot_examples(extraction_layers)
            all_few_samples.append({
                "all_xi_all_y_embeddings": all_xi_all_y_embeddings,
                "all_xi_yi_embeddings": all_xi_yi_embeddings,
                "few_shot_examples": few_shot_examples
            })
        # tt = time.time()

        # 并行化评估sample质量
        last_layer_name = f"layer_{self.model.layer_num}"
        assert last_layer_name in xq_embeddings.keys(), \
            f"last_layer_name: {last_layer_name} not in xq_embeddings.keys()"

        ## 取出所有的 all_xi_all_y_embeddings 组成batch
        batch_all_xi_all_y_embeddings = [
            sample["all_xi_all_y_embeddings"][last_layer_name] 
            for sample in all_few_samples
        ]
        batch_all_xi_all_y_embeddings = torch.stack(
            batch_all_xi_all_y_embeddings, dim=0
        ).to(self.model.device) # shape: (batch_size, num_x, num_y, K)

        ## 并行化计算 Xi 矩阵及其逆矩阵
        batch_Xi_matrix, batch_Xi_pinv = self.parallel_solve_Xi_matrix(
            batch_all_xi_all_y_embeddings
        )

        ## 并行化计算 lambda_1
        batch_lambda_1= self.parallel_solve_lambda_1_Xq_Xi_dagger(
            batch_Xi_pinv, xq_embeddings[last_layer_name].to(self.model.device)
        )

        ## 找到最优的 lambda_1
        optimal_index = int(torch.argmin(batch_lambda_1).cpu().item())

        # ttt = time.time()
        # print(f"评估所有样本耗时: {ttt - tt}")

        # 取得最优结果
        best_results = {
            "lambda_1": batch_lambda_1[optimal_index], 
            "few_shot_examples": all_few_samples[optimal_index]["few_shot_examples"], 
            "all_xi_all_y_embeddings": all_few_samples[optimal_index]["all_xi_all_y_embeddings"],
            "all_xi_yi_embeddings": all_few_samples[optimal_index]["all_xi_yi_embeddings"],
            "Xi_matrix": batch_Xi_matrix[optimal_index],
            "Xi_pinv": batch_Xi_pinv[optimal_index]
        }
        
        return best_results

    def evaluate_single_example(self, test_item: Dict[str, Any], extraction_layers: List[str]) -> Dict[str, Any]:
        """评估单个测试样例"""
        # 获得 \xi(x_Q)
        xq_embeddings, _ = self.sample_embeddings(test_item, extraction_layers)

        # 采样并评估few-shot quality
        now_time = time.time()

        last_layer_name = f"layer_{self.model.layer_num}"
        best_results = self.sample_and_evaluate_few_shot_quality(xq_embeddings, extraction_layers)
        
        Xi_pinv = best_results["Xi_pinv"].to(self.model.device)
        all_xi_yi_embeddings = best_results["all_xi_yi_embeddings"][last_layer_name].to(self.model.device)
        few_shot_examples = best_results["few_shot_examples"]
        lambda_1 = {last_layer_name: best_results["lambda_1"]}
        Xi_matrix = best_results["Xi_matrix"].to(self.model.device)
        rank = {last_layer_name: torch.linalg.matrix_rank(Xi_matrix)}

        # 计算每个层的 \bar{\xi}(x_i, y_i)
        mean_xi_yi_embeddings = torch.mean(all_xi_yi_embeddings, dim=0)

        # 计算 \alpha
        alpha = {last_layer_name: self.solve_alpha(Xi_pinv, mean_xi_yi_embeddings)}
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
