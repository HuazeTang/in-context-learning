model="qwen3-8b"
dataset="mmlu"
bash scripts/download_model_dataset.sh ${model} ${dataset}

method="random_prompt"
experiment_name="${model}_${dataset}_${method}"

subjects=(
    all
)

# 转换为Python列表格式
subjects_python=$(printf '"%s", ' "${subjects[@]}")
subjects_python="[${subjects_python%, }]"

mkdir -p log
python3 main.py evaluation=random_in_context model=qwen dataset.config.subjects="$subjects_python" | tee "log/${experiment_name}.txt"

if [ -f "scripts/s3upload.sh" ]; then
    bash scripts/s3upload.sh "log/${experiment_name}"
else
    echo "No s3upload.sh script found. Result will be saved locally."
fi
