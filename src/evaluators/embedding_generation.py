from tqdm import tqdm
from typing import List, Dict, Any
from torch import Tensor
from .information_in_context_evaluator import RandomInforInContextEvaluator
import os
import pickle

class EmbeddingGenerationEvaluator(RandomInforInContextEvaluator):
    def evaluate_single_example(self, test_item: Dict[str, Any], extraction_layers: List[str], pool_method: str) -> Dict[str, Any]:
        xq_embeddings, _ = self.sample_embeddings(test_item, extraction_layers, pool_method)
        xq_embeddings = xq_embeddings[0]
        xq_embeddings = {
            key: value.cpu() for key, value in xq_embeddings.items()
        }
        
        question = test_item['question']
        answer = test_item['answer']
        return {
            'question': question, 
            'answer': answer, 
            'choices': test_item['choices'],
            'embedding': xq_embeddings
        }
    
    def evaluate(self, output_dir: str = None):
        test_data = self.dataset.get_test_examples()
        dev_data = self.dataset.get_dev_examples()

        last_layer_name = f"layer_{self.model.layer_num}"
        extraction_layers = [last_layer_name]
        pool_method = self.config.get('pool_method', None)
        print("pool method: ", pool_method)

        # generate all embeddings
        all_test_embeddings: List[Dict[str, Tensor]] = []
        all_dev_embeddings: List[Dict[str, Tensor]] = []

        for test_item in tqdm(test_data):
            emb_result = self.evaluate_single_example(
                test_item, extraction_layers, pool_method
            )
            all_test_embeddings.append(emb_result)
        
        for dev_item in tqdm(dev_data):
            emb_result = self.evaluate_single_example(
                dev_item, extraction_layers, pool_method
            )
            all_dev_embeddings.append(emb_result)
        
        # save all embeddings
        all_embeddings = {
            'test': all_test_embeddings,
            'dev': all_dev_embeddings
        }

        if output_dir is None:
            output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'embeddings.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(all_embeddings, f)
        

        return 0.0, None, output_path
