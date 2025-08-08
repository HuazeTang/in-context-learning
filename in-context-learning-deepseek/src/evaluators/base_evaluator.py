import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from src.datasets import BaseDataset
from src.models import BaseModel
from src.utils import save_results

# keywords
## keywords for saving evaluation results
ACCURACY = "accuracy"
CORRECT_TIMES = "correct_times"
TOTAL_TIMES = "total_times"
RESULTS = "results"
CONFIG = "config"
DATASET_CONFIG = "dataset"
EVALUATION_CONFIG = "evaluation"

## keywords for prompt llm
ROLE = "role"
ROLE_SYSTEM = "system"
ROLE_USER = "user"
CONTENT = "content"

## keywords for creating result record
QUESTION = "question"
TRUE_ANSWER = "true_answer"
PRED_ANSWER = "pred_answer"
RESPONSE = "response"
IS_CORRECT = "is_correct"

# keywords
## keywords should be in the response
RESPONSE_MODEL = "response"
HIDDEN_STATES_MODEL = "hidden_states"



class BaseEvaluator(ABC):
    """评估器基类"""
    
    def __init__(self, model: BaseModel, dataset: BaseDataset, config: Dict, logger: logging.Logger):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.logger = logger
        
    @abstractmethod
    def evaluate(self, output_dir: str = None) -> Tuple[float, List[Dict[str, Any]], str]:
        """
        执行评估
        
        Args:
            output_dir: 结果保存目录
            
        Returns:
            Tuple[accuracy, results]: 准确率和详细结果列表
        """
        pass
    
    def save_evaluation_results(
            self, 
            accuracy: float, 
            correct: int, 
            total: int, 
            results: List[Dict[str, Any]], 
            output_dir: str
        ):
        """保存评估结果"""
        if output_dir:
            output_path = save_results({
                ACCURACY: accuracy,
                CORRECT_TIMES: correct,
                TOTAL_TIMES: total,
                RESULTS: results,
                CONFIG: {
                    DATASET_CONFIG: self.dataset.config,
                    EVALUATION_CONFIG: self.config
                }
            }, output_dir)
            return output_path
    
    def log_progress(self, current: int, total: int, correct: int):
        """记录评估进度"""
        if current % 10 == 0:
            accuracy = correct / current if current > 0 else 0
            self.logger.info(f"Processed {current}/{total} | Accuracy: {accuracy:.4f}")
    
    def log_final_results(self, accuracy: float, correct: int, total: int):
        """记录最终结果"""
        self.logger.info(f"Final Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    def prepare_messages(self, content: str, system_prompt: str = None) -> List[Dict[str, str]]:
        """准备模型输入消息"""
        messages = []
        if system_prompt:
            messages.append({ROLE: ROLE_SYSTEM, CONTENT: system_prompt})
        messages.append({ROLE: ROLE_SYSTEM, CONTENT: content})
        return messages
    
    def generate_response(self, messages: List[Dict[str, str]], extract_hidden_states: bool = False):
        """生成模型响应"""
        if extract_hidden_states:
            return self.model.generate(messages, return_hidden_states=True)
        else:
            return self.model.generate(messages)
    
    def evaluate_prediction(self, pred_answer: str, true_answer: str) -> bool:
        """评估预测结果"""
        return pred_answer == true_answer if pred_answer else False
    
    def create_result_record(
            self, 
            question: str, 
            true_answer: str, 
            pred_answer: str, 
            response: str, 
            is_correct: bool, 
            **kwargs
        ) -> Dict[str, Any]:
        """创建结果记录"""
        record = {
            QUESTION: question,
            TRUE_ANSWER: true_answer,
            PRED_ANSWER: pred_answer,
            RESPONSE: response,
            IS_CORRECT: is_correct
        }
        record.update(kwargs)
        return record