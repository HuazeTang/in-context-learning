from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_model import BaseModel
import torch
from torch import Tensor
from typing import Dict, Optional

THINK_TAG = "</think>"
THINK_TAG_INDEX = 151668 # this is the index of "</think>" in tokenizer

class QwenModel(BaseModel):
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map=self.config.device_map
        )
        self.device = self.model.device
        self.layer_num = self.config.layer_num
        
        # 添加终止符
        terminator_ids = []
        for terminator in self.config.terminators:
            if terminator in self.tokenizer.get_vocab():
                terminator_ids.append(self.tokenizer.convert_tokens_to_ids(terminator))
            elif terminator == self.tokenizer.eos_token:
                terminator_ids.append(self.tokenizer.eos_token_id)
        
        if terminator_ids:
            self.generation_params["eos_token_id"] = terminator_ids  
        
        # 设置思考模式
        self.enable_thinking = self.config.enable_thinking if hasattr(self.config, "enable_thinking") else False
    
    def get_embeddings(self, text: str, layers_to_extract: Optional[int] = None) -> Dict[str, Tensor]:
        """ 
        获取文本的逐层embedding
        
        Args:
            text: 输入文本
            layers_to_extract: 要提取的层索引列表，None表示提取所有层
            
        Returns:
            dict: 包含每层embedding的字典
        """
        # 对文本进行tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # 通过模型获取hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # 提取每层的hidden states
        embeddings = {}
        for layer_idx, layer_output in enumerate(outputs.hidden_states):
            if layers_to_extract is None or layer_idx in layers_to_extract:
                embeddings[f"layer_{layer_idx}"] = layer_output.cpu().detach()
        
        return {
            "embeddings": embeddings,
            "input_ids": inputs.input_ids.cpu(),
            "attention_mask": inputs.attention_mask.cpu() if "attention_mask" in inputs else None,
            "sequence_length": inputs.input_ids.shape[-1]
        }
    
    def generate(self, messages, return_hidden_states=False, layers_to_extract=None):
        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
        
        input_ids = self.tokenizer([text], return_tensors="pt",).to(self.device)

        # 设置生成参数
        generation_params = self.generation_params.copy()
        if return_hidden_states:
            generation_params["output_hidden_states"] = True
            generation_params["return_dict_in_generate"] = True
        
        # 生成响应
        outputs = self.model.generate(
            **input_ids,
            **generation_params
        )
        
        # 提取新生成的token
        if return_hidden_states:
            output_ids = outputs[0][len(input_ids.input_ids[0]):].tolist()
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            
            # 处理hidden states
            all_hidden_states = []
            for step_hidden_states in outputs.hidden_states:
                # step_hidden_states是一个tuple，包含所有层的输出
                step_layers = {}
                for layer_idx, layer_output in enumerate(step_hidden_states):
                    if layers_to_extract is None or layer_idx in layers_to_extract:
                        step_layers[f"layer_{layer_idx}"] = layer_output.cpu().detach()  # 移到CPU节省显存
                all_hidden_states.append(step_layers)
            
            return {
                "response": response,
                "hidden_states": all_hidden_states,  # 每个生成步骤的每一层输出
                "generated_ids": output_ids,
                "input_length": len(input_ids.input_ids[0])
            }
        else:
            output_ids = outputs[0][len(input_ids.input_ids[0]):].tolist() 
            # 解析思考内容（如果启用了思考模式）
            if self.enable_thinking:
                try:
                    # rindex finding 151668 (</think>)
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0
                
                thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            else:
                thinking_content = None
                content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            
            return {
                "response": content,
                "thinking": thinking_content,
            }