from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from datasets import Dataset

class BaseDataset(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self._load_data()
    
    @abstractmethod
    def _load_data(self):
        """加载数据集"""
        pass
    
    @abstractmethod
    def get_few_shot_examples(self, num_shots: int) -> List[Dict[str, Any]]:
        """获取few-shot示例"""
        pass
    
    @abstractmethod
    def get_dev_examples(self) -> Dataset:
        """获取训练集"""
        pass

    @abstractmethod
    def get_test_examples(self) -> Dataset:
        """获取测试集"""
        pass

    @abstractmethod
    def get_all_possible_answers(self, example: Dict[str, Any]) -> List[Union[str, int]]:
        """获取所有可能的答案"""
        pass

    @abstractmethod
    def get_ground_truth_index(self, example: Dict[str, Any]) -> Union[str, int]:
        """获取ground truth"""
        pass
    
    @abstractmethod
    def format_question(self, example: Dict[str, Any]) -> str:
        """格式化问题（包含选项）"""
        pass
    
    @abstractmethod
    def format_answer(self, example: Dict[str, Any]) -> str:
        """格式化答案"""
        pass
    
    @abstractmethod
    def extract_prediction(self, response: str) -> str:
        """从模型响应中提取预测"""
        pass