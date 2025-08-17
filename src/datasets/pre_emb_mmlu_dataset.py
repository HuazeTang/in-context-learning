from .base_dataset import BaseDataset
from typing import List, Dict, Any, Union
from .mmlu_dataset import MMLUDataset
import re
import os
import random
import pickle
import glob


class MMLUPreEmbDataset(MMLUDataset):
    def load_pickle(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    
    def load_subject_pickle(self, dataset_path, subject):
        if subject == "all":
            # load all subject data from local pkl file ends with "_results.pkl"
            file_pattern = f"./{dataset_path}/*.pkl"
            pkl_files = glob.glob(file_pattern)

            if not pkl_files:
                raise FileNotFoundError(f"No pkl files found matching pattern: {file_pattern}")
        
            # 合并所有subject的数据
            merged_data = {}
            for file_path in pkl_files:
                subject_data = self.load_pickle(file_path)
                
                # 初始化merged_data的结构
                if not merged_data:
                    merged_data = {key: [] for key in subject_data.keys()}
                
                # 合并每个split的数据
                for split_name, split_data in subject_data.items():
                    merged_data[split_name].extend(split_data)
            
            return merged_data
        else:
            file_path = f"./{dataset_path}/{subject}_results.pkl"
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
        return self.load_pickle(file_path)
    
    def _load_data(self):
        dataset_path = self.config.get('dataset_path', None)
        subject_list = self.config.get('subjects', None)
        test_datasets = []
        dev_datasets = []

        # load dataset from local pkl file
        # file path: f"./{datasetpath}/{subject}_results.pkl}", subject must be in file name
        
        for subj in subject_list:
            if dataset_path:
                data_all = self.load_subject_pickle(dataset_path, subj)
                test_data = data_all[self.config.get('test_split', 'test')]
                dev_data = data_all[self.config.get('dev_split', 'dev')]
            else:
                data_all = self.load_subject_pickle("cais/mmlu", subj)
                test_data = data_all[self.config.get('test_split', 'test')]
                dev_data = data_all[self.config.get('dev_split', 'dev')]
            
            # 为每条数据添加subject信息
            for item in test_data:
                item["subject"] = subj
            
            for item in dev_data:
                item["subject"] = subj
            
            test_datasets.extend(test_data)
            dev_datasets.extend(dev_data)
        
        self.test_data = test_datasets
        self.dev_data = dev_datasets
        
        # 限制样本数
        if max_samples := self.config.get('max_test_samples'):
            # 使用海象运算符来避免None值，同时进行赋值和条件判断
            if max_samples > 0:
                num_samples = min(max_samples, len(self.test_data))
                self.test_data = random.sample(self.test_data, num_samples)
    
    def get_few_shot_examples(self, num_shots: int) -> List[Dict[str, Any]]:
        num_examples = min(num_shots, len(self.dev_data))
        if num_examples == 0:
            return []
        
        # 使用datasets库的shuffle和select方法
        # seed = self.config.get('random_seed', 42)  # 默认种子保证可复现
        # shuffled_data = self.dev_data.shuffle(seed=seed)
        selected_data = random.sample(self.dev_data, num_examples)  
    
        return list(selected_data)

    def format_question(self, example: Dict[str, Any]) -> str:
        if "choices" in example:
            return super().format_question(example)
        else:
            # 直接返回问题
            question = example["question"]
            return f"Question: {question}"
