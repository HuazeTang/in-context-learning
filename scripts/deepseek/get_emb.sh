# 设置变量
model="deepseek-qwen3-8b"
dataset="mmlu"
bash scripts/download_model_dataset.sh ${model} ${dataset}
method="emb_gen"
pool_method="mean"

subjects=(
    "abstract_algebra"
    "anatomy"
    "astronomy"
    "business_ethics"
    "clinical_knowledge"
    "college_biology"
    "college_chemistry"
    "college_computer_science"
    "college_mathematics"
    "college_medicine"
    "college_physics"
    "computer_security"
    "conceptual_physics"
    "econometrics"
    "electrical_engineering"
    "elementary_mathematics"
    "formal_logic"
    "global_facts"
    "high_school_biology"
    "high_school_chemistry"
    "high_school_computer_science"
    "high_school_european_history"
    "high_school_geography"
    "high_school_government_and_politics"
    "high_school_macroeconomics"
    "high_school_mathematics"
    "high_school_microeconomics"
    "high_school_physics"
    "high_school_psychology"
    "high_school_statistics"
    "high_school_us_history"
    "high_school_world_history"
    "human_aging"
    "human_sexuality"
    "international_law"
    "jurisprudence"
    "logical_fallacies"
    "machine_learning"
    "management"
    "marketing"
    "medical_genetics"
    "miscellaneous"
    "moral_disputes"
    "moral_scenarios"
    "nutrition"
    "philosophy"
    "prehistory"
    "professional_accounting"
    "professional_law"
    "professional_medicine"
    "professional_psychology"
    "public_relations"
    "security_studies"
    "sociology"
    "us_foreign_policy"
    "virology"
    "world_religions"
)
mkdir -p log
mkdir -p log/all_results/

torch_dtype=float32

for subject in "${subjects[@]}"; do
    experiment_name="${model}_${dataset}-${subject}_${method}_${pool_method}"
    python3 main.py evaluation=emb_gen model=deepseek model.config.torch_dtype=$torch_dtype dataset.config.subjects="[$subject]" evaluation.config.pool_method="$pool_method" | tee "log/${experiment_name}.txt"

    result=$(cat "log/${experiment_name}.txt")
    output_file_path=$(echo $result | grep "Results saved to file:" | awk '{print $NF}')
    echo "output_file_path: $output_file_path"
    if [ -z "$output_file_path" ] || [ "$output_file_path" = "None" ]; then
        echo "Error: Failed to get output_file_path"
        exit 1
    fi
    echo "$output_file_path"

    if [ -n "$output_file_path" ] && [ "$output_file_path" != "None" ]; then
        # 生成新的文件名
        new_filename="${subject}_results.pkl"
        
        # 重命名并移动到目标目录
        mv "$output_file_path" "log/all_results/$new_filename"
        
        echo "File renamed and moved to: log/all_results/$new_filename"
    else
        echo "Error: Failed to get output_file_path for $subject"
        continue
    fi
done

tar -czf log/all_results.tar.gz -C log all_results/

s3_trained_model_folder=s3://uav-autotest-simulation-training/in_context_llm_prompting
date=$(date +%Y-%m-%d-%H-%M-%S)
s3_output_file_path=${s3_trained_model_folder}/${model}/${dataset}-${pool_method}-${torch_dtype}

s3cmd -c job/s3cfg put log/all_results.tar.gz ${s3_output_file_path}
