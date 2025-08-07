from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_model import BaseModel
import torch
from torch import Tensor
from typing import Dict, Optional, List, Union

class LLaMAModel(BaseModel):
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
        
        # 设置 pad_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        if terminator_ids:
            self.generation_params["eos_token_id"] = terminator_ids  
    
    @torch.inference_mode()
    def get_embeddings(self, texts: Union[str, List[str]], layers_to_extract: Optional[int] = None) -> Dict[str, Tensor]:
        """
        获取文本的逐层embedding
        
        Args:
            text: 输入文本
            layers_to_extract: 要提取的层索引列表，None表示提取所有层
            
        Returns:
            dict: 包含每层embedding的字典
        """
        # 对文本进行tokenize
        # 自动设置 batch size（可手动设置或做 chunk）
        MAX_BATCH_SIZE = 32 
        if isinstance(texts, str):
           texts = [texts]
        results = []

        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch_texts = texts[i:max(i + MAX_BATCH_SIZE, len(texts))]
            batch_size = len(batch_texts)
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

            # 通过模型获取hidden states
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            for b in range(batch_size):
                # 提取每层的hidden states
                embeddings = {}
                for layer_idx, layer_output in enumerate(outputs.hidden_states):
                    if layers_to_extract is None or layer_idx in layers_to_extract:
                        embeddings[f"layer_{layer_idx}"] = layer_output[b].unsqueeze(0).float()

                results.append({
                    "embeddings": embeddings,
                    "input_ids": inputs.input_ids[b].unsqueeze(0),
                    "attention_mask": inputs.attention_mask[b] if "attention_mask" in inputs else None,
                    "sequence_length": inputs.input_ids[b].shape[-1]
                })
        
        return results
    
    def generate(self, messages, return_hidden_states=False, layers_to_extract=None):
        # 应用聊天模板
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
        ).to(self.device)
        if isinstance(input_ids, dict):
            attention_mask = input_ids.get("attention_mask")
            input_ids = input_ids["input_ids"]
        else:
            attention_mask = None

        # 设置生成参数
        generation_params = self.generation_params.copy()
        generation_params["pad_token_id"] = self.tokenizer.pad_token_id

        if attention_mask is not None:
            generation_params["attention_mask"] = attention_mask

        if return_hidden_states:
            generation_params["output_hidden_states"] = True
            generation_params["return_dict_in_generate"] = True
        
        # 生成响应
        outputs = self.model.generate(
            input_ids,
            **generation_params
        )
        
        # 提取新生成的token
        if return_hidden_states:
            generated_ids = outputs.sequences[0][input_ids.shape[-1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
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
                "generated_ids": generated_ids,
                "input_length": input_ids.shape[-1]
            }
        else:
            generated_ids = outputs[0][input_ids.shape[-1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            return {
                "response": response,
                "hidden_states": None,
                "generated_ids": generated_ids,
                "input_length": input_ids.shape[-1]
            }
