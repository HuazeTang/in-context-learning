import os
from tqdm import tqdm
from typing import List, Dict, Optional, Union, Any
from .in_context_evaluator import RandomInContextEvaluator, RESPONSE_MODEL, HIDDEN_STATES_MODEL


class BatchInContextEvaluator(RandomInContextEvaluator):
    def construct_in_context_example(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build messages for the example"""
        num_shots = self.config.get('num_shots', 5)
        system_prompt = self.config.get('system_prompt')
        subject = example.get('subject', None)
        if subject is None:
            self.logger.warning("Subject not found in example. Few-shot examples will be drawn from all subjects.")
        
        # Build few-shot prompt
        few_shot_examples = self.dataset.get_few_shot_examples(num_shots, subject)
        few_shot_prompt = self.build_few_shot_prompt(few_shot_examples)
        prompt = self.build_full_prompt(example, few_shot_prompt)

        # Prepare messages with system prompt
        messages = self.prepare_messages(prompt, system_prompt)

        return messages, few_shot_prompt
    
    def evaluate(self, output_dir: str = None):
        """evalute in batch mode"""
        test_data = self.dataset.get_test_examples()

        # evaluation loop
        correct = 0.
        total = len(test_data)
        results = []
        batch_size = self.config.get('batch_size', 32)

        for i in tqdm(range(0, total, batch_size), desc="Evaluating"):
            max_index = min(i + batch_size, total)
            num_batch = max_index - i
            batch_items = test_data[i:max_index]
            messages_list = []
            few_shot_prompts = []

            for index in range(num_batch):
                test_item = {
                    k: v[index] for k, v in batch_items.items()
                }
                # build messages for each example
                messages, few_shot_prompt = self.construct_in_context_example(test_item)
                messages_list.append(messages)
                few_shot_prompts.append(few_shot_prompt)

            # generate results in batch
            batch_results = self.model.batch_generate(messages_list, return_hidden_states=self.config.get('extract_hidden_states', False))

            # process each result
            for j, result in enumerate(batch_results):
                assert RESPONSE_MODEL in result, f"Response not found in result: {result.keys()}"
                response = result[RESPONSE_MODEL]
                hidden_states = result.get(HIDDEN_STATES_MODEL, None)
                extract_hidden_states = self.config.get('extract_hidden_states', False)

                if not extract_hidden_states:
                    hidden_states = None

                # extract prediction and true answer
                test_item = {
                    k: v[j] for k, v in batch_items.items()
                }
                few_shot_prompt = few_shot_prompts[j]
                pred_answer = self.dataset.extract_prediction(response)
                true_answer = self.dataset.format_answer(test_item)

                result_record = self.create_result_record(
                    question=self.dataset.format_question(test_item),
                    true_answer=true_answer,
                    pred_answer=pred_answer,
                    response=response,
                    is_correct=self.evaluate_prediction(pred_answer, true_answer),
                    few_shot_prompt=few_shot_prompt,
                    hidden_states=hidden_states if extract_hidden_states else None
                )
                results.append(result_record)

                if result_record['is_correct']:
                    correct += 1

            # 定期记录进度
            self.log_progress(min(i + batch_size, total), total, correct)

        accuracy = correct / total if total > 0 else 0.0
        self.log_final_results(accuracy, correct, total)
        
        # 保存结果
        output_path = self.save_evaluation_results(accuracy, correct, total, results, output_dir)
        
        return accuracy, results, output_path
