from .llama_model import LLaMAModel
from typing import Dict, Optional, List, Union
import torch
import os
import logging
import json

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

class LLaMAVLLMModel(LLaMAModel):
    def __init__(self, config):
        super().__init__(config)
        if not VLLM_AVAILABLE:
            raise ImportError("VLLM is not available. Please install it with: pip install vllm")
    
    def load_model(self):
        """load VLLM model"""
        load_transformer = self.config.get('load_transformer', False)
        if load_transformer:
            # load transformers will cost more memory, only for embeddings extraction
            super().load_model()

        # initialize VLLM engine
        self.llm = LLM(
            model=self.config.model_path,
            tokenizer=self.config.tokenizer_path,
            dtype=self.config.torch_dtype,
            tensor_parallel_size=getattr(self.config, 'tensor_parallel_size', 1),
            gpu_memory_utilization=getattr(self.config, 'gpu_memory_utilization', 0.9),
            max_model_len=getattr(self.config, 'max_model_len', None),
            trust_remote_code=True,
            enforce_eager=True,
            disable_log_stats=True,
        )
        
        # obtain tokenizer (from VLLM)
        self.tokenizer = self.llm.get_tokenizer()
        tk_cfg = os.path.join(self.config.tokenizer_path, "tokenizer_config.json")
        if tpl := self.config.get('chat_template', None):
            self.tokenizer.chat_template = tpl
        elif getattr(self.tokenizer, "chat_template", None):
            pass
        elif os.path.exists(tk_cfg):
            with open(tk_cfg, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("chat_template"):
                self.tokenizer.chat_template = meta["chat_template"]
        else:
            raise ValueError(
                "No chat_template found in config, tokenizer_config.json, or fallback. "
                "Please set `config.chat_template` explicitly."
            )

        self.device = self.config.device_map
        self.layer_num = self.config.layer_num
        
        # set sampling parameters
        stops = list(self.config.get('terminators', None))
        assert isinstance(stops, List), f"Required stops be List, but get {type(stops)} instead"
        self.sampling_params = SamplingParams(
            temperature=self.generation_params.get('temperature', 0.6),
            top_p=self.generation_params.get('top_p', 0.9),
            max_tokens=self.generation_params.get('max_new_tokens', 512),
            stop=stops,
        )
    
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
            tokenize=False
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
                tokenize=False
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
    
    def get_embeddings(self, texts: Union[str, List[str]], layers_to_extract: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """VLLM does not support embeddings, fallback to transformers if available"""
        logging.warning("Warning: VLLM doesn't support embeddings extraction. This feature requires loading the model with transformers.")
        if not hasattr(self, 'model'):
            raise RuntimeError("Model not loaded with transformers. Please set 'load_transformer' to True in config.")
        return super().get_embeddings(texts, layers_to_extract)