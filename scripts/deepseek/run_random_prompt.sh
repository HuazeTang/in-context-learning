# 设置变量
model="deepseek-qwen3-8b"
dataset="mmlu"
bash scripts/download_model_dataset.sh ${model} ${dataset}

method="random_prompt"
experiment_name="${model}_${dataset}_${method}"

# 启动推理任务
mkdir -p log
python3 main.py evaluation=random_in_context model=deepseek | tee "log/${experiment_name}.txt"

# 可选上传结果
if [ -f "scripts/s3upload.sh" ]; then
    bash scripts/s3upload.sh "log/${experiment_name}"
else
    echo "No s3upload.sh script found. Result will be saved locally."
fi
