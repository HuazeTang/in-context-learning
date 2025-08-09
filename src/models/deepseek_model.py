from transformers import AutoTokenizer, AutoModelForCausalLM
from .base_model import BaseModel
import torch
from torch import Tensor
from typing import List, Dict, Optional
import os

class DeepSeekModel(BaseModel):
    def load_model(self):
        print(f"[DEBUG] Using DeepSeekModel with model_path = {self.config.model_path}")
        model_path = self.config.model_path
        tokenizer_path = self.config.tokenizer_path
        is_local = os.path.isdir(model_path)  # ✅ 判断是否是本地路径

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=is_local
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=getattr(torch, self.config.torch_dtype),
            device_map=self.config.device_map,
            trust_remote_code=is_local
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

    def get_embeddings(self, text: str, layers_to_extract: Optional[int] = None) -> Dict[str, Tensor]:
        """
        获取文本的逐层 embedding 表示

        Args:
            text: 输入文本
            layers_to_extract: 可选，指定要提取的层索引

        Returns:
            dict: 包含 embeddings、input_ids、attention_mask、sequence_length
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

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

        if return_hidden_states:
            generated_ids = outputs.sequences[0][input_ids.shape[-1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            all_hidden_states = []
            for step_hidden_states in outputs.hidden_states:
                step_layers = {}
                for layer_idx, layer_output in enumerate(step_hidden_states):
                    if layers_to_extract is None or layer_idx in layers_to_extract:
                        step_layers[f"layer_{layer_idx}"] = layer_output.cpu().detach()
                all_hidden_states.append(step_layers)

            return {
                "response": response,
                "hidden_states": all_hidden_states,
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
