import time
from .information_in_context_evaluator import RandomInforInContextEvaluator
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union
import torch


class RandomInforInContextEvaluatorPreEmb(RandomInforInContextEvaluator):
    def sample_embeddings(self, examples, extraction_layers, pool_method):
        r"""获得 \xi(x_Q)"""
        xq_embeddings_all = examples['embedding']
        return [{layer_name: xq_embeddings_all[layer_name] for layer_name in extraction_layers}], None

    def get_all_xi_all_y_embeddings(self, few_shot_examples, extraction_layers):
        r"""获得 few shot example 的 \xi(x,y)"""
        all_xi_all_y_embeddings = {}
        for layer_name in extraction_layers:
            random_picked_dev_tensor = torch.stack(
                [item['embedding'][layer_name] for item in few_shot_examples]
            )
            all_xi_all_y_embeddings[layer_name] = random_picked_dev_tensor
        
        return all_xi_all_y_embeddings
    
    def get_all_xi_yi_embeddings(self, few_shot_examples, extraction_layers):
        r"""获得 few shot example 的 \xi(x,y)"""
        all_xi_yi_embeddings = {}
        for layer_name in extraction_layers:
            random_picked_dev_tensor_list = []
            for item in few_shot_examples:
                answer = item['answer']
                random_picked_dev_tensor_list.append(item['embedding'][layer_name][answer])
            
            random_picked_dev_tensor = torch.stack(random_picked_dev_tensor_list)
            all_xi_yi_embeddings[layer_name] = random_picked_dev_tensor
        
        return all_xi_yi_embeddings
    
    def sample_few_shot_examples(self, extraction_layers, pool_method):
        r"""获得 few shot example 的 \xi(x,y)"""
        # 随机采样
        num_shots = self.config.get('num_shots', 5)
        few_shot_examples = self.dataset.get_few_shot_examples(num_shots)
        all_xi_all_y_embeddings = self.get_all_xi_all_y_embeddings(few_shot_examples, extraction_layers)
        all_xi_yi_embeddings = self.get_all_xi_yi_embeddings(few_shot_examples, extraction_layers)

        return all_xi_all_y_embeddings, all_xi_yi_embeddings, few_shot_examples
