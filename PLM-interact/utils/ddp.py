import os
import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


def ddp_setup():
    if 'SLURM_PROCID' in os.environ:
        os.environ['RANK'] = os.environ['SLURM_PROCID']
        rank  = int(os.environ['RANK'])
        gpus_per_node = int(os.environ['SLURM_GPUS_ON_NODE'])
        local_rank= rank - gpus_per_node * (rank // gpus_per_node)
        os.environ['LOCAL_RANK'] = str(local_rank)
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])

    # init_process_group(backend='nccl')
    ddp_rank       = int(os.environ['RANK'])        # Global rank for DDP.
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # Local rank for DDP.
    device         = f"cuda:{ddp_local_rank}"
    # torch.cuda.set_device(device)

    seed_offset = ddp_rank 

    return seed_offset,ddp_rank,ddp_local_rank,device



class SequentialDistributedSampler(Sampler):
    """
    from https://github.com/cpystan/Wsi-Caption/blob/master/modules/dataloaders.py

    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[
                  self.rank * self.num_samples: (self.rank + 1) * self.num_samples
                  ]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def distributed_concat(
        tensor,
        num_total_examples,
):
    output_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    dist.barrier()
    return concat[:num_total_examples]