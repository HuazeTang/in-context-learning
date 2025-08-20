from .base_dataset import BaseDataset
from datasets import load_dataset, concatenate_datasets
import random
import re
from typing import List, Dict, Any

class MMLUProDataset(BaseDataset):
    def _load_data(self):
        dataset_path = self.config.get('dataset_path', None)
        subject_list = self.config.get('subjects', None)
        test_datasets = []
        dev_datasets = []

        for subj in subject_list:
            if dataset_path:
                # 如果指定了本地路径，直接加载 default 再过滤
                raw_dataset = load_dataset(dataset_path, "default")
            else:
                # MMLU-Pro 只有 default 配置
                raw_dataset = load_dataset("TIGER-Lab/MMLU-Pro", "default")

            # 按 subject 过滤
            if "subject" in raw_dataset["test"].column_names:
                raw_dataset = raw_dataset.filter(lambda x: x["subject"] == subj)

            # 固定 validation 作为 dev 因为mmlu pro没有dev只有valid 和 test 用valid替代dev
            test_split_name = self.config.get('test_split', 'test')
            dev_split_name = 'validation'

            if test_split_name not in raw_dataset:
                raise ValueError(f"Test split '{test_split_name}' not found for subject {subj}")
            if dev_split_name not in raw_dataset:
                raise ValueError(f"Validation split '{dev_split_name}' not found for subject {subj}")

            test_data = raw_dataset[test_split_name]
            dev_data = raw_dataset[dev_split_name]

            test_datasets.append(test_data)
            dev_datasets.append(dev_data)

        # 合并所有学科
        self.test_data = concatenate_datasets(test_datasets)
        self.dev_data = concatenate_datasets(dev_datasets)

        # 限制样本数
        if max_samples := self.config.get('max_test_samples'):
            if max_samples < 0:
                self.test_data = self.test_data.select(range(len(self.test_data)))
            else:
                self.test_data = self.test_data.select(range(min(max_samples, len(self.test_data))))

    def get_few_shot_examples(self, num_shots: int):
        num_examples = min(num_shots, len(self.dev_data))
        if num_examples == 0:
            return []
        selected_indices = random.sample(range(len(self.dev_data)), num_examples)
        return list(self.dev_data.select(selected_indices))

    def get_dev_examples(self):
        return self.dev_data

    def get_test_examples(self):
        return self.test_data
# mmlu pro的数据格式{
#   "question": "...",
#   "options": ["A", "B", "C", "D"],
#   "answer": A,
#   "answer_index": 0,
#   "category": "engineering"
# }这里是options此外直接返回选项不必处理

    def get_all_possible_answers(self, example):
        return example["options"]

    def get_ground_truth_index(self, example):
        # 直接返回字母索引
        return example["answer_index"]

    def format_question(self, example):
        question = example["question"]
        options = "\n".join([f"{chr(65 + j)}. {opt}" for j, opt in enumerate(example["options"])])
        return f"Question: {question}\n{options}"

    def format_answer(self, example):
        # 直接返回字母
        return example["answer"]

    def extract_prediction(self, response: str):
        match = re.search(r'\b([A-D])\b', response)
        return match.group(1) if match else None
