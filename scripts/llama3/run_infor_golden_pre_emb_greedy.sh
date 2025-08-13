model="llama3-8b-no-download"
dataset="mmlu-per-emb-llama3-8b"
bash scripts/download_model_dataset.sh ${model} ${dataset}

method="infor-golden-pre-emb-greedy"
experiment_name="${model}_${dataset}_${method}"

mkdir -p log
python3 main.py evaluation=infor_golden_in_context_pre_emb_greedy dataset=mmlu_pre_emb_llama3 model=no_model model.config.layer_num=32 | tee "log/${experiment_name}.txt"

if [ -f "scripts/s3upload.sh" ]; then
    bash scripts/s3upload.sh "log/${experiment_name}"
else
    echo "No s3upload.sh script found. Result will be saved locally."
fi

experiment_name="${model}_${dataset}-float32_${method}"
python3 main.py evaluation=infor_golden_in_context_pre_emb_greedy dataset=mmlu_pre_emb_llama3_float32 model=no_model model.config.layer_num=32 | tee "log/${experiment_name}.txt"

if [ -f "scripts/s3upload.sh" ]; then
    bash scripts/s3upload.sh "log/${experiment_name}"
else
    echo "No s3upload.sh script found. Result will be saved locally."
fi
