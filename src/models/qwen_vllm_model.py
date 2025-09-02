from .llama_model import LLaMAModel
from typing import Dict, Optional, List, Union
import torch
import os
import logging
import json
from vllm import LLM, SamplingParams
from .llama_vllm_model import LLaMAVLLMModel


class QwenVLLMModel(LLaMAVLLMModel):
    def generate(self, messages, return_hidden_states=False, layers_to_extract=None):
        """apply VLLM for generation"""
        if return_hidden_states:
            # VLLM does not support hidden states, throw warning and fallback to transformers
            logging.warning("VLLM doesn't support hidden states extraction. Falling back to transformers.")
            return super().generate(messages, return_hidden_states, layers_to_extract)
        
        # apply chat templeta
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False
        )
        
        # VLLM generation
        outputs = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)
        response = outputs[0].outputs[0].text
        
        return {
            "response": response,
            "hidden_states": None,
            "generated_ids": None,
            "input_length": len(prompt)
        }
    
    def batch_generate(self, messages_list: List[List[Dict]], **kwargs):
        """Batch generation with VLLM"""
        prompts = []
        for messages in messages_list:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False
            )
            prompts.append(prompt)
        
        outputs = self.llm.generate(prompts, self.sampling_params)
        
        results = []
        for i, output in enumerate(outputs):
            results.append({
                "response": output.outputs[0].text,
                "hidden_states": None,
                "generated_ids": None,
                "input_length": len(prompts[i])
            })
        
        return results