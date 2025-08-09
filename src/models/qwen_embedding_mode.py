from transformers import AutoTokenizer, AutoModel
from .base_model import BaseModel
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Optional, List, Union

class QwenEmbModel(BaseModel):
    def load_model(self):
        # embedding 模型使用 left padding
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_path, 
            padding_side='left'
        )
        
        # 使用 AutoModel 而不是 AutoModelForCausalLM
        self.model = AutoModel.from_pretrained(
            self.config.model_path,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map=self.config.device_map,
            # 推荐启用 flash_attention_2
            attn_implementation="flash_attention_2" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else None
        )
        self.device = self.model.device
        self.layer_num = self.config.layer_num
        
        # 设置 pad_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """提取最后一个有效token的hidden state"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        """为查询添加指令"""
        return f'Instruct: {task_description}\nQuery:{query}'
    
    @torch.inference_mode()  # 更高效的推理模式
    def get_embeddings(
        self, 
        texts: Union[str, List[str]],
        layers_to_extract: Optional[List[int]] = None,
        normalize: bool = True,
        task_description: str = None,
        max_length: int = 8192
    ) -> List[Dict]:
        """ 
        获取文本的embedding（针对Qwen embedding模型优化）
        
        Args:
            texts: 输入文本
            layers_to_extract: 要提取的层索引列表（对于embedding模型通常只用最后一层）
            normalize: 是否对embedding进行L2标准化
            task_description: 任务描述（用于查询文本）
            max_length: 最大序列长度
            
        Returns:
            list: 包含embedding信息的字典列表
        """
        # A100 优化：更大的batch size
        MAX_BATCH_SIZE = 256  # A100 可以处理更大批次
        
        if isinstance(texts, str):
            texts = [texts]
        
        # 如果提供了task_description，为查询文本添加指令
        if task_description:
            texts = [self.get_detailed_instruct(task_description, text) for text in texts]
        
        results = []

        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch_texts = texts[i:max(i + MAX_BATCH_SIZE, len(texts))]
            batch_results = self._process_embedding_batch(
                batch_texts, layers_to_extract, normalize, max_length
            )
            results.extend(batch_results)
        
        return results

    def _process_embedding_batch(
            self, 
            batch_texts: List[str],
            layers_to_extract: Optional[List[int]],
            normalize: bool, 
            max_length: int
        ) -> List[Dict]:
        """处理单个batch的embedding"""
        batch_dict = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device, non_blocking=True)

        # 获取模型输出
        outputs = self.model(**batch_dict)
        
        batch_size = len(batch_texts)
        results = []
        
        for b in range(batch_size):
            embeddings = {}
            
            # 提取最后一个token的embedding（主要用途）
            last_hidden = self.last_token_pool(
                outputs.last_hidden_state[b:b+1], 
                batch_dict['attention_mask'][b:b+1]
            )
            
            if normalize:
                last_hidden = F.normalize(last_hidden, p=2, dim=1)
            
            embeddings["last_token"] = last_hidden.cpu().float()
            
            # 如果需要其他层的输出（可选）
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states and layers_to_extract:
                for layer_idx in layers_to_extract:
                    if layer_idx < len(outputs.hidden_states):
                        layer_output = self.last_token_pool(
                            outputs.hidden_states[layer_idx][b:b+1],
                            batch_dict['attention_mask'][b:b+1]
                        )
                        if normalize:
                            layer_output = F.normalize(layer_output, p=2, dim=1)
                        embeddings[f"layer_{layer_idx}"] = layer_output.cpu().float()
            
            results.append({
                "embeddings": embeddings,
                "input_ids": batch_dict['input_ids'][b].unsqueeze(0).cpu(),
                "attention_mask": batch_dict['attention_mask'][b].cpu(),
                "sequence_length": batch_dict['attention_mask'][b].sum().item()
            })
        
        # 清理GPU缓存
        del outputs
        torch.cuda.empty_cache()
        
        return results

    def compute_similarity(self, query_embeddings: List[Dict], 
                          doc_embeddings: List[Dict]) -> Tensor:
        """计算查询和文档之间的相似度分数"""
        query_embeds = torch.stack([item["embeddings"]["last_token"] for item in query_embeddings])
        doc_embeds = torch.stack([item["embeddings"]["last_token"] for item in doc_embeddings])
        
        # 计算余弦相似度
        scores = query_embeds @ doc_embeds.T
        return scores

    def generate(self, messages, return_hidden_states=False, layers_to_extract=None):
        """
        Embedding模型不支持生成功能
        """
        raise NotImplementedError("Embedding model does not support text generation")

# 使用示例的辅助函数
def example_usage():
    """演示如何使用改写后的模型"""
    # 假设config已经配置好
    # model = LLaMAModel(config)
    # model.load_model()
    
    # 任务描述
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    
    queries = [
        'What is the capital of China?',
        'Explain gravity'
    ]
    
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other."
    ]
    
    # 获取查询embeddings（带任务指令）
    # query_embeddings = model.get_embeddings(
    #     queries, 
    #     task_description=task,
    #     normalize=True
    # )
    
    # 获取文档embeddings（不需要任务指令）
    # doc_embeddings = model.get_embeddings(
    #     documents,
    #     normalize=True
    # )
    
    # 计算相似度
    # scores = model.compute_similarity(query_embeddings, doc_embeddings)
    # print(scores.tolist())