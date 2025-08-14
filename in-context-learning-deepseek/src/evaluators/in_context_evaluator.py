from tqdm import tqdm
from typing import List, Dict, Any
from .base_evaluator import BaseEvaluator, RESPONSE_MODEL, HIDDEN_STATES_MODEL



class RandomInContextEvaluator(BaseEvaluator):
    def build_few_shot_prompt(self, examples: List[Dict[str, Any]]) -> str:
        """构建few-shot提示"""
        prompt = ""
        for ex in examples:
            question = self.dataset.format_question(ex)
            answer = self.dataset.format_answer(ex)
            prompt += f"{question}\nAnswer: {answer}\n\n"
        return prompt.strip()
    
    def build_full_prompt(self, example: Dict[str, Any], few_shot_prompt: str) -> str:
        """构建完整提示"""
        return f"{few_shot_prompt}\n\n{self.dataset.format_question(example)}\nAnswer:"
    
    def evaluate_single_example(self, test_item: Dict[str, Any]) -> Dict[str, Any]:
        # 获取few-shot示例数量 / system prompt / extract_hidden_states 状态
        num_shots = self.config.get('num_shots', 5)
        system_prompt = self.config.get('system_prompt')
        extract_hidden_states = self.config.get('extract_hidden_states', False)
        
        # 构建完整提示
        few_shot_examples = self.dataset.get_few_shot_examples(num_shots)
        few_shot_prompt = self.build_few_shot_prompt(few_shot_examples)
        prompt = self.build_full_prompt(test_item, few_shot_prompt)

        # 准备消息
        messages = self.prepare_messages(prompt, system_prompt)
        
        # 生成答案
        result = self.generate_response(messages, extract_hidden_states=extract_hidden_states)
        assert RESPONSE_MODEL in result, f"Response not found in result: {result.keys()}"
        response = result[RESPONSE_MODEL]

        if extract_hidden_states:
            assert HIDDEN_STATES_MODEL in result, f"Hidden states not found in result: {result.keys()}"
            hidden_states = result[HIDDEN_STATES_MODEL]
        else:
            hidden_states = None
        
        # 提取预测
        pred_answer = self.dataset.extract_prediction(response)
        true_answer = self.dataset.format_answer(test_item)

        return {
            'true_answer': true_answer,
            'pred_answer': pred_answer,
            'response': response,
            'few_shot_prompt': few_shot_prompt,
            'hidden_states': hidden_states if extract_hidden_states else None
        }

    
    def evaluate(self, output_dir: str = None):
        """执行评估"""
        
        # 获取测试集
        test_data = self.dataset.get_test_examples()
        
        # 评估循环
        correct = 0
        total = len(test_data)
        results = []
        
        for i, test_item in enumerate(tqdm(test_data, desc="Evaluating")):
            test_result = self.evaluate_single_example(test_item)
            true_answer = test_result['true_answer']
            pred_answer = test_result['pred_answer']
            response = test_result['response']
            few_shot_prompt = test_result['few_shot_prompt']
            hidden_states = test_result['hidden_states']
            
            # 评估结果
            is_correct = self.evaluate_prediction(pred_answer, true_answer)
            if is_correct:
                correct += 1
            
            # 记录结果
            result_record = self.create_result_record(
                question=self.dataset.format_question(test_item),
                true_answer=true_answer,
                pred_answer=pred_answer,
                response=response,
                is_correct=is_correct,
                few_shot_prompt=few_shot_prompt,
                hidden_states=hidden_states
            )
            results.append(result_record)
            
            # 定期记录进度
            self.log_progress(i + 1, total, correct)
        
        accuracy = correct / total
        self.log_final_results(accuracy, correct, total)
        
        # 保存结果
        output_path = self.save_evaluation_results(accuracy, correct, total, results, output_dir)
        
        return accuracy, results, output_path