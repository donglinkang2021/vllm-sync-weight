# Minimal Sync Weight Example for vLLM

Minimal ray+vllm example that demonstrates how to synchronize model weights

```bash
HF_HUB_OFFLINE=1 python rlhf.py
```

A server-client architecture is used where the server hosts the vLLM model for inference, and the client performs RLHF training and updates the model weights on the server. (Modified from the [trl.extras.vllm_client](https://github.com/huggingface/trl/blob/main/trl/extras/vllm_client.py) and [trl.scripts.vllm_serve](https://github.com/huggingface/trl/blob/main/trl/scripts/vllm_serve.py).)

Run the following command to start the server(inference/rollout):

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python -m vllm_sync.server model=Qwen/Qwen2.5-Math-1.5B
```

Then run the client(training) script:

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python vllm_demo.py
```