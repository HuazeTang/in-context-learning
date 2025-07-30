if [ -f "scripts/download_model_dataset_hope.sh" ]; then
    echo "Download with hope"
    bash scripts/download_model_dataset_hope.sh
else   
    mkdir -p data/datasets
    mkdir -p data/model_ckpts

    mkdir -p data/model_ckpts/meta-llama
    git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct data/model_ckpts/meta-llama/Meta-Llama-3-8B-Instruct

    mkdir -p data/model_ckpts/Qwen
    git clone https://huggingface.co/Qwen/Qwen3-8B data/model_ckpts/Qwen/Qwen3-8B
    mv huggingface.co/Qwen/ data/model_ckpts

    mkdir -p data/datasets/cais/mmlu
    git clone https://huggingface.co/datasets/cais/mmlu data/datasets/cais/mmlu
fi
