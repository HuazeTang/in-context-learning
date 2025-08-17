from .base_dataset import BaseDataset
from datasets import load_dataset
import random
import re
from typing import List, Dict, Any


class GPQADataset(BaseDataset):
    def _load_data(self):
        """
        加载 GPQA 数据集 (gpqa_main/gpqa_extended/gpqa_diamond/gpqa_experts)，
        将 train 划分为 dev/test
        """
        dataset_path = self.config.get('dataset_path', None)
        config_name = self.config.get('config_name', 'gpqa_main')  # 可选: gpqa_main, gpqa_extended, gpqa_diamond, gpqa_experts
        split_ratio = self.config.get('dev_ratio', 0.25)  # dev 集比例

        # 加载整个 train split
        if dataset_path:
            dataset_all = load_dataset(dataset_path, config_name, split="train")
        else:
            dataset_all = load_dataset("Idavidrein/gpqa", config_name, split="train")

        dataset_all = dataset_all.shuffle(seed=42)  # 固定随机种子保证可复现

        # 按比例划分 dev/test
        dev_size = int(split_ratio * len(dataset_all))
        self.dev_data = dataset_all.select(range(dev_size))
        self.test_data = dataset_all.select(range(dev_size, len(dataset_all)))

        # 限制测试集样本数
        if max_samples := self.config.get('max_test_samples'):
            if max_samples < 0:
                self.test_data = self.test_data.select(range(len(self.test_data)))
            else:
                self.test_data = self.test_data.select(
                    range(min(max_samples, len(self.test_data)))
                )

    def get_few_shot_examples(self, num_shots: int) -> List[Dict[str, Any]]:
        """
        从 dev 集中随机抽取 few-shot 示例
        """
        num_examples = min(num_shots, len(self.dev_data))
        if num_examples == 0:
            return []
        selected_indices = random.sample(range(len(self.dev_data)), num_examples)
        return list(self.dev_data.select(selected_indices))

    def get_dev_examples(self) -> List[Dict[str, Any]]:
        return self.dev_data

    def get_test_examples(self):
        return self.test_data

    def get_all_possible_answers(self, example: Dict[str, Any]) -> List[str]:
        """
        GPQA-main 的选项字段：Correct Answer + Incorrect Answer 1~3
        """
        if all(
            k in example
            for k in ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
        ):
            return [
                example["Correct Answer"],
                example["Incorrect Answer 1"],
                example["Incorrect Answer 2"],
                example["Incorrect Answer 3"],
            ]
        else:
            raise ValueError(f"No valid choice fields found in example: {example}")

    def get_ground_truth_index(self, example: Dict[str, Any]) -> int:
        """
        正确答案是 Correct Answer 在选项列表中的位置（固定为索引 0）
        """
        return 0

    def format_question(self, example: Dict[str, Any]) -> str:
        """
        格式化问题
        """
        question = example["Question"]
        options = self.get_all_possible_answers(example)
        options_str = "\n".join([f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)])
        return f"Question: {question}\n{options_str}"

    def format_answer(self, example: Dict[str, Any]) -> str:
        """
        格式化答案（字母形式）
        """
        return chr(65 + self.get_ground_truth_index(example))

    def extract_prediction(self, response: str) -> str:
        """
        从模型响应中提取预测的答案字母 (A, B, C, D)
        """
        match = re.search(r'\b([A-D])\b', response)
        return match.group(1) if match else None
