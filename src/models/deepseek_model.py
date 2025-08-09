from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_model import BaseModel
from .llama_model import LLaMAModel
import torch
from torch import Tensor
from typing import List, Dict, Optional
import os
import re

class DeepSeekModel(LLaMAModel):
    def load_model(self):
        model_path = os.path.abspath(self.config.model_path)
        tokenizer_path = os.path.abspath(self.config.tokenizer_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        print(f"[DEBUG] Using DeepSeekModel with model_path = {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map=self.config.device_map,
            trust_remote_code=False
        )

        self.device = self.model.device
        self.layer_num = self.config.layer_num

        # 设置 pad_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 添加终止符
        terminator_ids = []
        for terminator in self.config.terminators:
            if terminator in self.tokenizer.get_vocab():
                terminator_ids.append(self.tokenizer.convert_tokens_to_ids(terminator))
            elif terminator == self.tokenizer.eos_token:
                terminator_ids.append(self.tokenizer.eos_token_id)

        if terminator_ids:
            self.generation_params["eos_token_id"] = terminator_ids

    def generate(self, messages, return_hidden_states=False, layers_to_extract=None):
        """
        执行多轮对话生成

        Args:
            messages: 多轮对话列表（格式符合 apply_chat_template 的输入）
            return_hidden_states: 是否返回中间层表示
            layers_to_extract: 可选，指定要提取的层索引

        Returns:
            dict: 包含 response、hidden_states、generated_ids、input_length 等
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
        ).to(self.device)

        attention_mask = input_ids.get("attention_mask") if isinstance(input_ids, dict) else None
        input_ids = input_ids["input_ids"] if isinstance(input_ids, dict) else input_ids

        generation_params = self.generation_params.copy()
        generation_params["pad_token_id"] = self.tokenizer.pad_token_id
        if attention_mask is not None:
            generation_params["attention_mask"] = attention_mask
        if return_hidden_states:
            generation_params["output_hidden_states"] = True
            generation_params["return_dict_in_generate"] = True    

        outputs = self.model.generate(input_ids, **generation_params)
        assert return_hidden_states is False, "DeepSeek model does not support return_hidden_states"
        
        generated_ids = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        thinking_process, final_response = self._parse_thinking_output(response)
        # print(f"[DEBUG] full response = {response}, final_response = {final_response}")
        return {
            "response": final_response,
            "thinking_process": thinking_process,
            "hidden_states": None,
            "generated_ids": generated_ids,
            "input_length": input_ids.shape[-1]
        }

    def _parse_thinking_output(self, raw_output: str) -> tuple[str, str]:
        """
        解析 DeepSeek thinking model 的输出，分离思考过程和最终回答
        
        Args:
            raw_output: 模型的原始输出
            
        Returns:
            tuple: (thinking_process, final_response)
        """
        
        # 尝试匹配 ... 标签
        # 匹配 <think>...</think> 标签
        if "<think>" in raw_output and "</think>" not in raw_output:
            raise Warning(f"Invalid output format: {raw_output}")
        
        thinking_pattern = r'<think>(.*?)</think>'
        thinking_match = re.search(thinking_pattern, raw_output, re.DOTALL)

        if thinking_match:
            # 提取思考过程
            thinking_process = thinking_match.group(1).strip()
            
            # 提取最终回答（移除thinking标签部分）
            final_response = re.sub(thinking_pattern, '', raw_output, flags=re.DOTALL).strip()
        
        else:
            # 如果没有找到thinking标签，将整个输出作为最终回答
            
            thinking_process = ""
            final_response = raw_output.strip()
        
        return thinking_process, final_response