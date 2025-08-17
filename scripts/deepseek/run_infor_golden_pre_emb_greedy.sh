model="llama3-8b-no-download"
dataset="mmlu-per-emb-deepseek-qwen3-8b"
bash scripts/download_model_dataset.sh ${model} ${dataset}

method="infor-golden-pre-emb-greedy"
experiment_name="${model}_${dataset}_${method}"
base_dir="./data/datasets/deepseek-qwen3-8b-mmlu"

mkdir -p log

run_experiment() {
    local name_suffix=$1
    local dataset_path_arg=$2
    
    local experiment_name="${model}_${dataset}${name_suffix}_${method}_${repeat_time}"
    
    # run the python3 cmd and save the output to a log file
    python3 main.py evaluation=infor_golden_in_context_pre_emb_greedy dataset=mmlu_pre_emb_deepseek_qwen3 model=no_model model.config.layer_num=32 dataset.config.dataset_path=${dataset_path_arg} | tee "log/${experiment_name}.txt"

    # upload results
    if [ -f "scripts/s3upload.sh" ]; then
        bash scripts/s3upload.sh "log/${experiment_name}"
    else
        echo "No s3upload.sh script found. Result will be saved locally."
    fi
}

# run last pool method with float16
run_experiment "-last-float16" "${base_dir}/all_results_last_float16"

# run last pool method with float32
run_experiment "-last-float32" "${base_dir}/all_results_last_float32"

# run last pool method with float64
run_experiment "-last-float64" "${base_dir}/all_results_last_float64"

# run mean pool method with float16
run_experiment "" "${base_dir}/all_results"

# run mean pool method with float32
run_experiment "-float32" "${base_dir}/all_results_float32"

# run mean pool method with float64
run_experiment "-float64" "${base_dir}/all_results_float64"