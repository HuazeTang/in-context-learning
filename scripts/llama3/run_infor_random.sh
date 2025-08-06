model="llama3-8b"
dataset="mmlu"
bash scripts/download_model_dataset.sh ${model} ${dataset}

method="infor_random"
experiment_name="${model}_${dataset}_${method}"

python3 main.py evaluation=infor_random_in_context | tee $experiment_name.txt

if [ -f "scripts/s3upload.sh" ]; then
    bash scripts/s3upload.sh ${experiment_name}
else
    echo "No s3upload.sh script found. Result will be saved locally."
fi
