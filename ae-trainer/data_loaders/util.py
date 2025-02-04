import torch

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # Configure this worker's shard
    dataset.num_shards = worker_info.num_workers
    dataset.shard_id = worker_info.id
