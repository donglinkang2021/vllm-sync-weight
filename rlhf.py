# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning from human feedback (RLHF) using vLLM and Ray.

The script separates training and inference workloads onto distinct GPUs
so that Ray can manage process placement and inter-process communication.
A Hugging Face Transformer model occupies GPU 0 for training, whereas a
tensor-parallel vLLM inference engine occupies GPU 1.

The example performs the following steps:

* Load the training model on GPU 0.
* Split the inference model across GPUs 1 using vLLM's tensor parallelism
  and Ray placement groups.
* Generate text from a list of prompts using the inference engine.
* Update the weights of the training model and broadcast the updated weights
  to the inference engine by using a Ray collective RPC group. Note that
  for demonstration purposes we simply zero out the weights.

For a production-ready implementation that supports multiple training and
inference replicas, see the OpenRLHF framework:
https://github.com/OpenRLHF/OpenRLHF

This example assumes a single-node cluster with two GPUs, but Ray
supports multi-node clusters. vLLM expects the GPUs are only used for vLLM
workloads. Residual GPU activity interferes with vLLM memory profiling and
causes unexpected behavior.
"""

import os

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rlhf_utils import get_communicator
from transformers import AutoModelForCausalLM
from typing import Optional, List

from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port
from vllm.outputs import RequestOutput

class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)

class RayLLM:
    """Wrapper class that combines Ray placement groups with vLLM LLM."""
    
    def __init__(
        self,
        model: str,
        gpu_ids: Optional[List[int]] = None,
        tensor_parallel_size: int = 1,
        **llm_kwargs
    ):
        """
        Initialize RayLLM with placement group management.
        
        Args:
            model: Model name or path
            gpu_ids: Specific GPU IDs to use (e.g., [1])
            tensor_parallel_size: Tensor parallel size for vLLM
            **llm_kwargs: Additional arguments passed to vLLM LLM
        """
        # Set GPU visibility if specified
        if gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        # Initialize Ray
        ray.init()
        
        num_gpus = len(gpu_ids) if gpu_ids is not None else torch.cuda.device_count()
        # Create placement group
        self.pg = placement_group([{"GPU": 1, "CPU": 0}] * num_gpus)
        ray.get(self.pg.ready())
        
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=self.pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        
        # Launch remote LLM
        self._remote_llm = ray.remote(
            num_cpus=0,
            num_gpus=0,
            scheduling_strategy=scheduling_strategy,
        )(MyLLM).remote(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="ray",
            **llm_kwargs
        )
    
    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        **kwargs
    ) -> List[RequestOutput]:
        """Generate text from prompts."""
        return ray.get(self._remote_llm.generate.remote(prompts, sampling_params, **kwargs))
    
    def collective_rpc(self, method: str, args=None, kwargs=None):
        """Call collective RPC method on the remote LLM."""
        args = () if args is None else args        
        kwargs = {} if kwargs is None else kwargs
        return self._remote_llm.collective_rpc.remote(method, args=args, **kwargs)
    
    def __getattr__(self, name):
        """Forward other attribute access to remote LLM."""
        def method(*args, **kwargs):
            return ray.get(getattr(self._remote_llm, name).remote(*args, **kwargs))
        return method


class Client:
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.communicator = None
        self.host = get_ip()
        self.port = get_open_port()

    def init_communicator(self, device: torch.device):
        handle = llm.collective_rpc(
            "init_communicator",
            args=(self.host, self.port, 1, world_size)
        )
        self.communicator = get_communicator(
            self.host, self.port, 0, world_size, device
        )
        ray.get(handle)
    
    def update_weight(self, llm:RayLLM, model: torch.nn.Module):
        for name, p in model.named_parameters():
            handle = llm.collective_rpc("update_weight", args=(name, p.dtype, p.shape))
            self.communicator.broadcast(p, src=0, stream=torch.cuda.current_stream())
            ray.get(handle)


# Load the training model
# train_model = AutoModelForCausalLM.from_pretrained(
#     "Qwen/Qwen2.5-Math-1.5B-Instruct", dtype="float16"
# ).to("cuda:0")
model0 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B", dtype="float16"
).to("cuda:0")
model1 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B", dtype="float16"
).to("cuda:0")
model2 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Math-1.5B-Instruct", dtype="float16"
).to("cuda:0")

# Create RayLLM with clean API
llm = RayLLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    gpu_ids=[1,2],  # Use GPU 1 for inference
    tensor_parallel_size=2,
    enforce_eager=True,
    worker_extension_cls="rlhf_utils.WorkerExtension",
)

world_size = 3  # 2 workers + 1 trainer
client = Client(world_size=world_size)
client.init_communicator(device=torch.device("cuda:0"))

# Generate text - now with proper type hints and IDE support
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)

print("-" * 50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)

client.update_weight(llm, model0)

# Generate with updated model
outputs_updated = llm.generate(prompts, sampling_params)
print("-" * 50)
for output in outputs_updated:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)

client.update_weight(llm, model1)

# Generate with updated model
outputs_updated = llm.generate(prompts, sampling_params)
print("-" * 50)
for output in outputs_updated:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)

client.update_weight(llm, model2)

# Generate with updated model
outputs_updated = llm.generate(prompts, sampling_params)
print("-" * 50)
for output in outputs_updated:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)