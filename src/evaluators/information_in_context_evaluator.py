import time
import torch
from torch import Tensor
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union
from .base_evaluator import BaseEvaluator, RESPONSE_MODEL, HIDDEN_STATES_MODEL
from enum import Enum
from collections import defaultdict


class RandomInforInContextEvaluator(BaseEvaluator):
    def build_prompt_with_answer(self, question: str, answer: str):
        return f"{question}\nAnswer: {answer}\n\n"
    
    def build_few_shot_prompt(self, examples: List[Dict[str, Any]]) -> str:
        """构建few-shot提示"""
        prompt = ""
        for ex in examples:
            question = self.dataset.format_question(ex)
            answer = self.dataset.format_answer(ex)
            prompt += self.build_prompt_with_answer(question, answer)
        return prompt.strip()
    
    def pool_embedding(self, hidden_state: torch.Tensor, method: str = "mean") -> torch.Tensor:
        """对embedding进行池化"""
        assert len(hidden_state.shape) == 3, "hidden_state must be 3D tensor"
        assert hidden_state.shape[0] == 1, "batch size for hidden state must be 1"
        
        if method == "mean":
            return hidden_state.mean(dim=1)
        elif method == "max":
            return hidden_state.max(dim=1)[0]
        elif method == "first":
            return hidden_state[:, 0, :]
        elif method == "last":
           return hidden_state[:, -1, :]
        else:
            raise ValueError(f"Invalid pooling method: {method}")

    def get_embedding(self, text: Union[str, List[str]], extraction_layers: List[str], pool_method: str="mean") -> torch.Tensor:
        """获取文本的embedding"""
        embeddings = self.model.get_embeddings(text) # 并行获取所有 embeddings
        assert isinstance(embeddings, list), f"Invalid type for embeddings: {type(embeddings)}"
        pool_hidden_state = []
        
        for single_embedding in embeddings:
            hidden_state = single_embedding["embeddings"]
            single_pool_hidden_state = {}
            for k, v in hidden_state.items():
                if k not in extraction_layers:
                    continue
                single_pool_hidden_state[k] = self.pool_embedding(v, pool_method)
            pool_hidden_state.append(single_pool_hidden_state)

        return pool_hidden_state

    def collect_embeddings(self, current_example_batch_embeddings: List[Dict[str, torch.Tensor]], true_y_index: int):
        xi_all_y_text_embeddings = defaultdict(list)
        for xi_y_text_embedding in current_example_batch_embeddings:
            for layer_name, embedding in xi_y_text_embedding.items():
                assert len(embedding.shape) > 1, f"Invalid shape for embedding: {embedding.shape}"
                xi_all_y_text_embeddings[layer_name].append(embedding)
        
        # 转换为普通字典并处理embeddings
        xi_all_y_text_embeddings = dict(xi_all_y_text_embeddings)
        xi_yi_embeddings = {}

        for layer_name, embeddings_list in xi_all_y_text_embeddings.items():
            # stack所有embeddings并标准化
            emb_all_y = torch.cat(embeddings_list, dim=0)  # shape: (num_y, K)

                # 标准化处理
            emb_all_y = emb_all_y - torch.mean(emb_all_y, dim=0, keepdim=True)
            norms = torch.norm(emb_all_y, dim=0, keepdim=True)
            norms = torch.where(norms == 0, torch.ones_like(norms), norms) # 避免除零
            xi_all_y_text_embeddings[layer_name] = emb_all_y / norms

            # get yi embedding
            xi_yi_embeddings[layer_name] = xi_all_y_text_embeddings[layer_name][true_y_index]
        
        return xi_all_y_text_embeddings, xi_yi_embeddings
    
    def sample_embeddings(
            self, 
            examples: Union[List[Dict[str, Any]], Dict[str, Any]], 
            extraction_layers: List[str], 
            pool_method: str = "mean"
        ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        """采样示例并输出归一化后的embeddings"""
        if isinstance(examples, dict):
            examples = [examples]
        
        all_xi_all_y_text_embeddings, all_xi_yi_embeddings = [], []
        
        for example in examples:
            question = self.dataset.format_question(example)
            all_possible_y = self.dataset.get_all_possible_answers(example)

            assert len(all_possible_y) > 0, f"No possible answers for example: {example}"

            true_y_index = self.dataset.get_ground_truth_index(example)
            assert true_y_index is not None, \
                f"Invalid ground truth index for example: {example}. \
                Maybe the answer for the dataset is not countable."

            # 构建并行prompt
            prompts = [self.build_prompt_with_answer(question, y).strip() for y in all_possible_y]
            # 并行获取所有 embeddings
            batch_embeddings = self.get_embedding(prompts, extraction_layers, pool_method) 

            xi_all_y_text_embeddings, xi_yi_embeddings = self.collect_embeddings(
                batch_embeddings, true_y_index
            )

            all_xi_all_y_text_embeddings.append(xi_all_y_text_embeddings)
            all_xi_yi_embeddings.append(xi_yi_embeddings)
        
        return all_xi_all_y_text_embeddings, all_xi_yi_embeddings

    def sample_few_shot_examples(self, extraction_layers: List[str], pool_method: str = "mean") -> List[Dict[str, Any]]:
        """采样few-shot并输出归一化后的embeddings"""
        num_shots = self.config.get('num_shots', 5)

        few_shot_examples = self.dataset.get_few_shot_examples(num_shots)

        all_xi_all_y_embeddings = defaultdict(list)
        all_xi_yi_embeddings = defaultdict(list)
        all_xi_all_y_text_embeddings_, all_xi_yi_embeddings_ = self.sample_embeddings(
            few_shot_examples, extraction_layers, pool_method
        )
        for xi_all_y_text_embeddings_, xi_yi_embeddings_ in zip(all_xi_all_y_text_embeddings_, all_xi_yi_embeddings_):
            for k in xi_all_y_text_embeddings_.keys():
                all_xi_all_y_embeddings[k].append(xi_all_y_text_embeddings_[k])
                all_xi_yi_embeddings[k].append(xi_yi_embeddings_[k])

        # 转换为普通字典并处理embeddings
        all_xi_all_y_embeddings = dict(all_xi_all_y_embeddings)
        all_xi_yi_embeddings = dict(all_xi_yi_embeddings)

        for k in all_xi_all_y_embeddings.keys():
            all_xi_all_y_embeddings[k] = torch.stack(all_xi_all_y_embeddings[k], dim=0)
            all_xi_yi_embeddings[k] = torch.stack(all_xi_yi_embeddings[k], dim=0)
        
        return all_xi_all_y_embeddings, all_xi_yi_embeddings, few_shot_examples
    
    def solve_Xi_matrix(self, all_xi_all_y_layer_emb: Tensor) -> Tensor:
        """求解 Xi matrix"""
        assert len(all_xi_all_y_layer_emb.shape) == 3, f"Invalid shape for all_xi_all_y_layer_emb: {all_xi_all_y_layer_emb.shape}"
        
        num_x, num_y, K = all_xi_all_y_layer_emb.shape
        
        # Xi_matrix = torch.zeros((K, K), device=all_xi_all_y_layer_emb.device, dtype=all_xi_all_y_layer_emb.dtype)
        reshaped_emb = all_xi_all_y_layer_emb.reshape(num_x * num_y, K).float()
        Xi_matrix = reshaped_emb.T @ reshaped_emb / num_x
        # check whether Xi_matrix is hermitian matrix
        assert torch.allclose(Xi_matrix, Xi_matrix.T), "Xi_matrix is not hermitian matrix"
        Xi_pinv = torch.linalg.pinv(Xi_matrix, hermitian=True)
        
        return Xi_matrix, Xi_pinv
    
    def solve_alpha(self, Xi_pinv: Tensor, mean_xi_yi_layer_emb: Tensor) -> Tensor:
        """求解 alpha 参数"""
        assert len(mean_xi_yi_layer_emb.shape) == 1, f"Invalid shape for mean_xi_yi_layer_emb: {mean_xi_yi_layer_emb.shape}"
        
        K = Xi_pinv.shape[-1]
        assert mean_xi_yi_layer_emb.shape == (K,), f"Invalid shape for mean_xi_yi_layer_emb: {mean_xi_yi_layer_emb.shape}"
    
        # 使用 lstsq 代替直接求逆，更快且数值稳定（弃用）
        # alpha = torch.linalg.lstsq(Xi_matrix.float(), mean_xi_yi_layer_emb.float()).solution
        # 直接使用逆矩阵乘法
        alpha = Xi_pinv @ mean_xi_yi_layer_emb.float()
        
        return alpha

    def solve_lambda_1_Xq_Xi_dagger(self, Xi_pinv: Tensor, xq_embeddings: Tensor) -> Tensor:
        """求解 \lambda_1 (\Xi(x_Q) \Xi^\dagger)"""
        # 对每一个 y 求解 xq * xq^\top * \Xi^\dagger
        assert len(xq_embeddings.shape) == 2, f"Invalid shape for xq_embeddings: {xq_embeddings.shape}"
        num_y, K = xq_embeddings.shape
        assert Xi_pinv.shape == (K, K), f"Invalid shape for Xi_pinv: {Xi_pinv.shape}, should be (f{K}, f{K})"
        
        Xq_matrix = xq_embeddings.float().T @ xq_embeddings.float()
        Xq_Xi_dagger_matrix = Xq_matrix @ Xi_pinv
        
        # 求解 Xq_Xi_dagger_matrix 的最大特征值
        lambda_1 = torch.linalg.eigvalsh(Xq_Xi_dagger_matrix.float()).max()
        return lambda_1
    
    def solve_metrics(self, all_xi_all_y_layer_emb: Tensor, mean_xi_yi_layer_emb: Tensor, xq_embeddings: Tensor) -> Dict[str, Tensor]:
        all_xi_all_y_layer_emb = all_xi_all_y_layer_emb.to(self.model.device)
        mean_xi_yi_layer_emb = mean_xi_yi_layer_emb.to(self.model.device)
        xq_embeddings = xq_embeddings.to(self.model.device)
        
        Xi_matrix, Xi_pinv = self.solve_Xi_matrix(all_xi_all_y_layer_emb)

        # jump over rank calculation
        # rank = torch.linalg.matrix_rank(Xi_matrix, tol=1e-2)
        rank = torch.tensor(-1)

        alpha = self.solve_alpha(Xi_pinv, mean_xi_yi_layer_emb)
        lambda_1 = self.solve_lambda_1_Xq_Xi_dagger(Xi_pinv, xq_embeddings)

        return {
            "alpha": alpha,
            "lambda_1": lambda_1,
            "rank": rank
        }
    
    def compute_predictions(self, xq_embeddings: Dict[str, Tensor], alpha: Dict[str, Tensor]) -> Dict[str, int]:
        """计算预测结果"""
        hat_P = {
            k: torch.matmul(xq_embeddings[k].float(), alpha[k]) 
            for k in alpha.keys()
        }
        
        argmax_hat_P = {
            k: torch.argmax(hat_P[k]).cpu()
            for k in hat_P.keys()
        }
        
        return argmax_hat_P
    
    def evaluate_single_example(self, test_item: Dict[str, Any], extraction_layers: List[str], pool_method: str) -> Dict[str, Any]:
        """评估单个测试样例"""
        # 获得 \xi(x_Q)
        xq_embeddings, _ = self.sample_embeddings(test_item, extraction_layers, pool_method)
        xq_embeddings = xq_embeddings[0]
        
        # 获得 few shot example 的 \xi(x,y)
        (
            all_xi_all_y_embeddings, 
            all_xi_yi_embeddings, 
            few_shot_examples
        ) = self.sample_few_shot_examples(extraction_layers, pool_method)

        # 计算每个层的 \bar{\xi}(x_i, y_i)
        mean_xi_yi_embeddings = {k: torch.mean(v, dim=0) for k, v in all_xi_yi_embeddings.items()}
        
        # 求解 \hat{\alpha} = \Xi^\dagger \bar{\xi}(x_i, y_i)
        now_time = time.time()
        alpha = {}
        lambda_1 = {}
        rank = {}
        for k in all_xi_all_y_embeddings.keys():
            if k not in extraction_layers:
                continue
            
            metrics = self.solve_metrics(all_xi_all_y_embeddings[k], mean_xi_yi_embeddings[k], xq_embeddings[k])
            alpha[k] = metrics['alpha']
            lambda_1[k] = metrics['lambda_1'].cpu()
            rank[k] = metrics['rank'].cpu()
        
        solve_time = time.time() - now_time

        # 计算预测
        argmax_hat_P = self.compute_predictions(xq_embeddings, alpha)
        
        return {
            'predictions': argmax_hat_P,
            'few_shot_examples': few_shot_examples,
            'solve_time': solve_time,
            'lambda_1': lambda_1,
            'rank': rank
        }

    def evaluate(self, output_dir: str = None) -> Tuple[float, List[Dict[str, Any]]]:
        """执行评估"""
        # 获取测试集
        test_data = self.dataset.get_test_examples()

        # 评估循环
        correct = 0
        total = len(test_data)
        results = []
        correct_all = dict()

        assert hasattr(self.model, "layer_num"), f"Model must have attribute 'layer_num'"
        last_layer_name = f"layer_{self.model.layer_num}"
        extraction_layers = [last_layer_name]
        pool_method = self.config.get('pool_method', None)
        print("pool method: ", pool_method)

        for i, test_item in enumerate(tqdm(test_data, desc="Evaluating")):
            single_case_result = self.evaluate_single_example(test_item, extraction_layers, pool_method)
            argmax_hat_P = single_case_result['predictions']
            few_shot_examples = single_case_result['few_shot_examples']
            lambda_1 = single_case_result['lambda_1'][last_layer_name].item()
            rank = single_case_result['rank'][last_layer_name].item()
            response = single_case_result.get('response', 'none')
            
            true_answer = self.dataset.format_answer(test_item)
            pred_answer = chr(65+argmax_hat_P[last_layer_name].cpu().item())

            for k in argmax_hat_P.keys():
                if k not in correct_all:
                    correct_all[k] = 0
                
                pred_answer_k = chr(65+argmax_hat_P[k].cpu().item())
                is_correct_k = self.evaluate_prediction(pred_answer_k, true_answer)
                if is_correct_k:
                    correct_all[k] += 1
            
            current_correct_string = ""
            for k,v in correct_all.items():
                acc = v / (i+1)
                current_correct_string += f"{k}: {acc:.2f}; "
            self.logger.info(f"{i+1}/{total}: {current_correct_string}")

            # 评估结果
            is_correct = self.evaluate_prediction(pred_answer, true_answer)
            if is_correct:
                correct += 1
            
            # 记录结果
            result_record = self.create_result_record(
                question=self.dataset.format_question(test_item),
                true_answer=true_answer,
                pred_answer=pred_answer,
                response=response,
                is_correct=is_correct,
                few_shot_prompt=self.build_few_shot_prompt(few_shot_examples),
                hidden_states=None,
            )
            result_record["lambda_1"] = lambda_1
            result_record["rank"] = rank
            results.append(result_record)
            
            # 定期记录进度
            self.log_progress(i + 1, total, correct)
        
        accuracy = correct / total
        self.log_final_results(accuracy, correct, total)
        
        # 保存结果
        output_path = self.save_evaluation_results(accuracy, correct, total, results, output_dir)
        
        return accuracy, results, output_path
