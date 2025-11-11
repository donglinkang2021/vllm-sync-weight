import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

def get_communicator(
    master_address: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> PyNcclCommunicator:
    """
    Initialize a stateless process group for distributed communication.

    Args:
        master_address: The address of the master node.
        master_port: The port number of the master node.
        rank: The rank of the current process.
        world_size: The total number of processes.
        device: The device (e.g., GPU) to be used for communication.

    Returns:
        A PyNcclCommunicator for distributed communication.
    """
    pg = StatelessProcessGroup.create(
        master_address, master_port, rank, world_size)
    return PyNcclCommunicator(pg, device=device)

class WorkerExtension:
    """
    The class for vLLM's worker to inherit from.
    By defining an extension class, the code can work no matter what is
    the underlying worker class. This way, the code can be compatible
    with both vLLM V0 and V1.
    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """

    def init_communicator(self, master_address, master_port, rank_offset, world_size):
        from vllm.distributed.parallel_state import get_world_group
        rank = get_world_group().rank + rank_offset
        self.communicator = get_communicator(
            master_address, master_port, rank, world_size, self.device)

    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.communicator.broadcast(
            weight, src=0, stream=torch.cuda.current_stream())
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight

    def close_communicator(self):
        if self.communicator is not None:
            del self.communicator
            self.communicator = None
