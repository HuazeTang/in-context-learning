import time
import torch
from torch import Tensor
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from .information_in_context_golden_example_evaluator import InformationInContextGoldenExampleEvaluator


class InformationInContextExampleCompareEvaluator(InformationInContextGoldenExampleEvaluator):
    def sample_and_evaluate_few_shot_quality(self, xq_embeddings: Dict[str, Tensor], extraction_layers: List[str], pool_method: str):
        golden_examples_sample_times = self.config.get("golden_examples_sample_times", 10)

        # 一次性采样所有的 samples
        all_few_samples = []
        # t = time.time()
        for _ in range(golden_examples_sample_times):
            (
                all_xi_all_y_embeddings, 
                all_xi_yi_embeddings, 
                few_shot_examples
            ) = self.sample_few_shot_examples(extraction_layers, pool_method)
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
        few_shot_examples = dict()
        all_xi_all_y_embeddings_examples = dict()
        all_xi_yi_embeddings_examples = dict()
        for i in range(len(all_few_samples)):
            few_shot_examples[i] = all_few_samples[i]["few_shot_examples"]
            all_xi_all_y_embeddings_examples[i] = all_few_samples[i]["all_xi_all_y_embeddings"]
            all_xi_yi_embeddings_examples[i] = all_few_samples[i]["all_xi_yi_embeddings"]

        # 返回所有结果
        results = {
            "lambda_1": batch_lambda_1, 
            "few_shot_examples": few_shot_examples, 
            "all_xi_all_y_embeddings": all_xi_all_y_embeddings_examples,
            "all_xi_yi_embeddings": all_xi_yi_embeddings_examples,
            "Xi_matrix": batch_Xi_matrix,
            "Xi_pinv": batch_Xi_pinv
        }
        
        return results

    def evaluate_single_example(self, test_item: Dict[str, Any], extraction_layers: List[str], pool_method: str) -> Dict[str, Any]:
        """评估单个测试样例"""
        # 获得 \xi(x_Q)
        xq_embeddings, _ = self.sample_embeddings(test_item, extraction_layers, pool_method)
        xq_embeddings = xq_embeddings[0]

        # 采样并评估few-shot quality
        now_time = time.time()

        last_layer_name = f"layer_{self.model.layer_num}"
        all_example_results = self.sample_and_evaluate_few_shot_quality(xq_embeddings, extraction_layers, pool_method)

        golden_examples_sample_times = self.config.get("golden_examples_sample_times", 10)

        all_results = []
        best_argmax_hat_P = None
        best_lambda_1 = None
        best_rank = None
        best_few_shot_examples = None
        for i in range(golden_examples_sample_times):
            Xi_pinv = all_example_results["Xi_pinv"][i].to(self.model.device)
            all_xi_yi_embeddings = all_example_results["all_xi_yi_embeddings"][i][last_layer_name].to(self.model.device)
            few_shot_examples = all_example_results["few_shot_examples"][i]
            lambda_1 = {last_layer_name: all_example_results["lambda_1"][i]}
            Xi_matrix = all_example_results["Xi_matrix"][i].to(self.model.device)
            rank = {last_layer_name: torch.linalg.matrix_rank(Xi_matrix)}

            # 计算每个层的 \bar{\xi}(x_i, y_i)
            mean_xi_yi_embeddings = torch.mean(all_xi_yi_embeddings, dim=0)

            # 计算 \alpha
            alpha = {last_layer_name: self.solve_alpha(Xi_pinv, mean_xi_yi_embeddings)}
            xq_embeddings_cuda = {k: v.to(self.model.device) for k, v in xq_embeddings.items()}

            argmax_hat_P_cuda = self.compute_predictions(xq_embeddings_cuda, alpha)
            argmax_hat_P = {k: v.cpu() for k, v in argmax_hat_P_cuda.items()}

            all_results.append({
                'predictions': argmax_hat_P,
                'few_shot_examples': few_shot_examples,
                'lambda_1': lambda_1,
                'rank': rank
            })

            # 找到最好的结果
            if best_argmax_hat_P is None:
                best_argmax_hat_P = argmax_hat_P
                best_lambda_1 = lambda_1
                best_rank = rank
                best_few_shot_examples = few_shot_examples
            elif lambda_1[last_layer_name] < best_lambda_1[last_layer_name]:
                best_argmax_hat_P = argmax_hat_P
                best_lambda_1 = lambda_1
                best_rank = rank
                best_few_shot_examples = few_shot_examples
        
        solve_time = time.time() - now_time
        
        return {
            'predictions': best_argmax_hat_P,
            'few_shot_examples': best_few_shot_examples,
            'solve_time': solve_time,
            'lambda_1': best_lambda_1,
            'rank': best_rank,
            'all_results': all_results
        }

    def evaluate(self, output_dir: str = None) -> Tuple[float, List[Dict[str, Any]]]:
        """执行评估"""
        # 获取测试集
        test_data = self.dataset.get_test_examples()

        # 评估循环
        correct = 0
        total = len(test_data)
        results = []

        assert hasattr(self.model, "layer_num"), f"Model must have attribute 'layer_num'"
        last_layer_name = f"layer_{self.model.layer_num}"
        extraction_layers = [last_layer_name]
        pool_method = self.config.get("pool_method", None)

        for i, test_item in enumerate(tqdm(test_data, desc="Evaluating")):
            true_answer = self.dataset.format_answer(test_item)
            single_case_result = self.evaluate_single_example(test_item, extraction_layers, pool_method)
            single_case_all_result = single_case_result['all_results']

            best_argmax_hat_P = single_case_result['predictions']
            for example_result in single_case_all_result:
                argmax_hat_P = example_result['predictions']
                few_shot_examples = example_result['few_shot_examples']
                lambda_1 = example_result['lambda_1'][last_layer_name].item()
                rank = example_result['rank'][last_layer_name].item()

                pred_answer = chr(65+argmax_hat_P[last_layer_name].cpu().item())
                is_correct = self.evaluate_prediction(pred_answer, true_answer)
                
                # 记录结果
                result_record = self.create_result_record(
                    question=self.dataset.format_question(test_item),
                    true_answer=true_answer,
                    pred_answer=pred_answer,
                    response="none",
                    is_correct=is_correct,
                    few_shot_prompt=self.build_few_shot_prompt(few_shot_examples),
                    hidden_states=None,
                )
                result_record["lambda_1"] = lambda_1
                result_record["rank"] = rank
                results.append(result_record)

            best_pred_answer = chr(65+best_argmax_hat_P[last_layer_name].cpu().item())  
            is_best_correct = self.evaluate_prediction(best_pred_answer, true_answer)    
            if is_best_correct:
                correct += 1
            
            acc = correct / (i+1)
            self.logger.info(f"{i+1}/{total}: {acc:.4f}")
            
            # 定期记录进度
            self.log_progress(i + 1, total, correct)
        
        accuracy = correct / total
        self.log_final_results(accuracy, correct, total)
        
        # 保存结果
        output_path = self.save_evaluation_results(accuracy, correct, total, results, output_dir)
        
        return accuracy, results, output_path