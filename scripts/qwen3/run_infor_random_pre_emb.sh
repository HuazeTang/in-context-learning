model="qwen3-8b-no-dowload"
dataset="mmlu-per-emb-qwen3-8b"
bash scripts/download_model_dataset.sh ${model} ${dataset}

method="infor_random_pre_emb"
experiment_name="${model}_${dataset}_${method}"

mkdir -p log

for repeat_time in 1 2 3 4 5; do
    experiment_name="${model}_${dataset}_${method}_${repeat_time}"
    python3 main.py evaluation=infor_random_in_context_pre_emb dataset=mmlu_pre_emb_qwen3 model=no_model model.config.layer_num=36 | tee "log/${experiment_name}.txt"

    if [ -f "scripts/s3upload.sh" ]; then
        bash scripts/s3upload.sh "log/${experiment_name}"
    else
        echo "No s3upload.sh script found. Result will be saved locally."
    fi
done

for repeat_time in 1 2 3 4 5; do
    experiment_name="${model}_${dataset}-float32_${method}_${repeat_time}"
    python3 main.py evaluation=infor_random_in_context_pre_emb dataset=mmlu_pre_emb_qwen3_float32 model=no_model model.config.layer_num=36 | tee "log/${experiment_name}.txt"

    if [ -f "scripts/s3upload.sh" ]; then
        bash scripts/s3upload.sh "log/${experiment_name}"
    else
        echo "No s3upload.sh script found. Result will be saved locally."
    fi
done