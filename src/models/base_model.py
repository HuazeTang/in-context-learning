from typing import Dict
from abc import ABC, abstractmethod

# keywords


class BaseModel(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.generation_params = config.get('generation_params', {})
        self.layer_num = config.get('layer_num', -1)

        self.load_model()
        
    # @abstractmethod
    def load_model(self):
        pass

    # @abstractmethod
    def get_embeddings(self, text):
        pass
    
    # @abstractmethod
    def generate(self, messages):
        pass
    
    def format_messages(self, messages):
        """将通用消息格式转换为模型特定的输入格式"""
        return messages
    
    def to_device(self, tensor):
        """将张量移动到模型所在的设备"""
        if self.device:
            return tensor.to(self.device)
        return tensor