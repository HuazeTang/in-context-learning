model="llama3-8b"
dataset="mmlu-per-emb-llama3-8b"
bash scripts/download_model_dataset.sh ${model} ${dataset}

method="infor_random"
experiment_name="${model}_${dataset}_${method}"

mkdir -p log
python3 main.py evaluation=infor_random_in_context dataset=mmlu_pre_emb_llama3 | tee "log/${experiment_name}.txt"

if [ -f "scripts/s3upload.sh" ]; then
    bash scripts/s3upload.sh "log/${experiment_name}"
else
    echo "No s3upload.sh script found. Result will be saved locally."
fi
