from .base_dataset import BaseDataset
from datasets import load_dataset, concatenate_datasets
from typing import List, Dict, Any, Union, Optional
import re
import random

class MMLUDataset(BaseDataset):
    def _load_data(self):
        dataset_path = self.config.get('dataset_path', None)
        subject_list = self.config.get('subjects', None)
        test_datasets = []
        dev_datasets = []
        val_datasets = []
        train_datasets = []
        
        for subj in subject_list:
            if dataset_path:
                test_data = load_dataset(dataset_path, subj)[self.config.get('test_split', 'test')]
                dev_data = load_dataset(dataset_path, subj)[self.config.get('dev_split', 'dev')]
                val_data = load_dataset(dataset_path, subj)[self.config.get('val_split', 'validation')]
                train_keyword = self.config.get('train_split', 'auxiliary_train')
                if train_keyword in load_dataset(dataset_path, subj):
                    train_data = load_dataset(dataset_path, subj)[self.config.get('train_split', 'auxiliary_train')]
                else:
                    train_data = None
            else:
                test_data = load_dataset("cais/mmlu", subj)[self.config.get('test_split', 'test')]
                dev_data = load_dataset("cais/mmlu", subj)[self.config.get('dev_split', 'dev')]
                val_data = load_dataset("cais/mmlu", subj)[self.config.get('val_split', 'validation')]
                train_keyword = self.config.get('train_split', 'auxiliary_train')
                if train_keyword in load_dataset("cais/mmlu", subj):
                    train_data = load_dataset("cais/mmlu", subj)[self.config.get('train_split', 'auxiliary_train')]
                else:
                    train_data = None
            
            # add subject column
            # maybe do not need, as subject is already in the data
            if 'subject' not in test_data.column_names and subj != 'all':
                test_data = test_data.add_column("subject", [subj] * len(test_data))
                dev_data = dev_data.add_column("subject", [subj] * len(dev_data))
                val_data = val_data.add_column("subject", [subj] * len(val_data))
            
            test_datasets.append(test_data)
            dev_datasets.append(dev_data)
            val_datasets.append(val_data)
            if train_data:
                train_datasets.append(train_data)
        
        self.test_data = concatenate_datasets(test_datasets)
        self.dev_data = concatenate_datasets(dev_datasets)
        self.val_data = concatenate_datasets(val_datasets)
        if train_datasets:
            self.train_data = concatenate_datasets(train_datasets)
        else:
            self.train_data = []
        
        # limit the number of test samples if specified
        if max_samples := self.config.get('max_test_samples'):
            # use the walrus operator to avoid None value, and do assignment and condition check at the same time
            if max_samples < 0:
                self.test_data = self.test_data.select(range(len(self.test_data)))
            else:
                self.test_data = self.test_data.select(range(min(max_samples, len(self.test_data))))
    
    def get_few_shot_examples(self, num_shots: int, subj: Optional[str]=None) -> List[Dict[str, Any]]:
        if subj is not None:
            all_dev_data = self.dev_data.filter(lambda x: x['subject'] == subj)
        else:
            all_dev_data = self.dev_data
        total_samples = len(all_dev_data)
        num_examples = min(num_shots, total_samples)
        if num_examples == 0:
            return []
        
        # 使用datasets库的select方法
        selected_indices = random.sample(range(total_samples), num_examples)
        selected_data = all_dev_data.select(selected_indices)
    
        return list(selected_data)
    
    def get_dev_examples(self) -> List[Dict[str, Any]]:
        return self.dev_data

    def get_test_examples(self):
        return self.test_data

    def get_validation_examples(self):
        return self.val_data

    def get_train_examples(self):
        return self.train_data
    
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