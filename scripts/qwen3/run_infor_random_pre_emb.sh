model="qwen3-8b-no-dowload"
dataset="mmlu-per-emb-qwen3-8b"
bash scripts/download_model_dataset.sh ${model} ${dataset}

method="infor_random_pre_emb"
experiment_name="${model}_${dataset}_${method}"
base_dir="./data/datasets/qwen3-mmlu"

mkdir -p log

run_experiment() {
    local repeat_time=$1
    local name_suffix=$2
    local dataset_path_arg=$3
    
    local experiment_name="${model}_${dataset}${name_suffix}_${method}_${repeat_time}"
    
    # run the python3 cmd and save the output to a log file
    python3 main.py evaluation=infor_random_in_context_pre_emb dataset=mmlu_pre_emb_qwen3 model=no_model model.config.layer_num=36 dataset.config.dataset_path=${dataset_path_arg} | tee "log/${experiment_name}.txt"

    # upload results
    if [ -f "scripts/s3upload.sh" ]; then
        bash scripts/s3upload.sh "log/${experiment_name}"
    else
        echo "No s3upload.sh script found. Result will be saved locally."
    fi
}

# run last pool method with float16
for repeat_time in 1 2 3 4 5; do
    run_experiment ${repeat_time} "-last-float16" "${base_dir}/all_results_last_float16"
done

# run last pool method with float32
for repeat_time in 1 2 3 4 5; do
    run_experiment ${repeat_time} "-last-float32" "${base_dir}/all_results_last_float32"
done

# run last pool method with float64
for repeat_time in 1 2 3 4 5; do
    run_experiment ${repeat_time} "-last-float64" "${base_dir}/all_results_last_float64"
done

# run mean pool method with float16
for repeat_time in 1 2 3 4 5; do
    run_experiment ${repeat_time} "" "${base_dir}/all_results"
done

# run mean pool method with float32
for repeat_time in 1 2 3 4 5; do
    run_experiment ${repeat_time} "-float32" "${base_dir}/all_results_float32"
done

# run mean pool method with float64
for repeat_time in 1 2 3 4 5; do
    run_experiment ${repeat_time} "-float64" "${base_dir}/all_results_float64"
done