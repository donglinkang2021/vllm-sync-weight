export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1
uv run vllm_server.py --model Qwen/Qwen2.5-Math-1.5B