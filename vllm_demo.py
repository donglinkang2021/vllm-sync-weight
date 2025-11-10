from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm_client import VLLMClient

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")

client = VLLMClient()
client.init_communicator()

def generate_text():
    """Generate text using the VLLM client."""
    # Generate completions
    output = client.generate(
        ["Hello, AI!", "Tell me a joke"], 
        n=4, max_tokens=32, temperature=0.6, top_p=0.95
    )
    # print("Responses:", responses)  # noqa
    print("====================== Generated Text ====================")
    for completion_ids in output["completion_ids"]:
        # completion_ids is list[int] of token IDs
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        print("Generated text:", text)

generate_text()

# Update model weights
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct").to("cuda")
client.update_model_params(model)

generate_text()

# HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python vllm_demo.py