from .base_dataset import BaseDataset
from datasets import load_dataset
from typing import List, Dict, Any, Union
import re

class MMLUDataset(BaseDataset):
    def _load_data(self):
        dataset_path = self.config.get('dataset_path', None)
        subject = self.config.get('subject', 'high_school_biology')
        if dataset_path:
            self.test_data = load_dataset(dataset_path, subject)[self.config.get('test_split', 'test')]
            self.dev_data = load_dataset(dataset_path, subject)[self.config.get('dev_split', 'dev')]
        else:
            self.test_data = load_dataset("cais/mmlu", subject)[self.config.get('test_split', 'test')]
            self.dev_data = load_dataset("cais/mmlu", subject)[self.config.get('dev_split', 'dev')]
        
        # 限制样本数
        if max_samples := self.config.get('max_test_samples'):
            # 使用海象运算符来避免None值，同时进行赋值和条件判断
            if max_samples < 0:
                self.test_data = self.test_data.select(range(len(self.test_data)))
            else:
                self.test_data = self.test_data.select(range(min(max_samples, len(self.test_data))))
    
    def get_few_shot_examples(self, num_shots: int) -> List[Dict[str, Any]]:
        num_examples = min(num_shots, len(self.dev_data))
        if num_examples == 0:
            return []
        
        # 使用datasets库的shuffle和select方法
        seed = self.config.get('random_seed', 42)  # 默认种子保证可复现
        shuffled_data = self.dev_data.shuffle(seed=seed)
        selected_data = shuffled_data.select(range(num_examples))
    
        return list(selected_data)
    
    def get_dev_examples(self) -> List[Dict[str, Any]]:
        return self.dev_data

    def get_test_examples(self):
        return self.test_data
    
    def get_all_possible_answers(self, example: Dict[str, Any]) -> List[str]:
        return ["A", "B", "C", "D"]
    
    def get_ground_truth_index(self, example: Dict[str, Any]) -> Union[str, int]:
        return example["answer"]

    def format_question(self, example: Dict[str, Any]) -> str:
        question = example["question"]
        # chr(65+0) = 'A'
        # chr(65+1) = 'B'
        # chr(65+2) = 'C'
        # chr(65+3) = 'D'
        choices = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(example["choices"])])
        return f"Question: {question}\n{choices}"
    
    def format_answer(self, example: Dict[str, Any]) -> str:
        return chr(65 + example["answer"])
    
    def extract_prediction(self, response: str) -> str:
        """从模型响应中提取答案字母 (A, B, C, D)"""
        match = re.search(r'\b([A-D])\b', response)
        return match.group(1) if match else None