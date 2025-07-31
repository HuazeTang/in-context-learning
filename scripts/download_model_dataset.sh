if [ -f "scripts/download_model_dataset_hope.sh" ]; then
    echo "Download with hope"
    bash scripts/download_model_dataset_hope.sh
else   
    mkdir -p data/datasets
    mkdir -p data/model_ckpts

    if [ ! -d "data/model_ckpts/meta-llama/Meta-Llama-3-8B-Instruct" ]; then
        mkdir -p data/model_ckpts/meta-llama
        git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct data/model_ckpts/meta-llama/Meta-Llama-3-8B-Instruct
    else
        echo "Meta-Llama-3-8B-Instruct already exists"
    fi

    if [ ! -d "data/model_ckpts/Qwen/Qwen3-8B" ]; then
        mkdir -p data/model_ckpts/Qwen
        git clone https://huggingface.co/Qwen/Qwen3-8B data/model_ckpts/Qwen/Qwen3-8B data/model_ckpts/Qwen/Qwen3-8B
    else
        echo "Qwen3-8B already exists"
    fi

    if [ ! -d "data/datasets/cais/mmlu" ]; then
        mkdir -p data/datasets/cais/mmlu
        git clone https://huggingface.co/datasets/cais/mmlu data/datasets/cais/mmlu
    else
        echo "mmlu already exists"
    fi
fi
