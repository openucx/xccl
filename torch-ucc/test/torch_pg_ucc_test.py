import os
import sys
import argparse
import torch
import torch.distributed as dist
import torch_ucc

parser = argparse.ArgumentParser(description="Process Group UCC test")
parser.add_argument("--backend", type=str, default='mpi')
parser.add_argument("--op", type=str, default='p2p')
parser.add_argument("--use-cuda", type=bool, default=False)
args = parser.parse_args()

try:
    size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
except:
    print('OMPI env variables are not found')
    sys.exit(1)

os.environ['MASTER_PORT'] = '32167'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['RANK']        = str(rank)
os.environ['WORLD_SIZE']  = str(size)


torch.cuda.set_device(rank)
print("World size {}, rank {}".format(size, rank))
dist.init_process_group(args.backend, rank=rank, world_size=size)

t = torch.zeros([size]) + rank
t2 = torch.zeros([size])
use_cuda = args.use_cuda and torch.cuda.is_available()

if (args.backend == "nccl") or use_cuda:
    print("Using cuda tensor")
    t = t.cuda()

if args.op == "p2p":
    if rank == 0:
        dist.send(t, 1)
    else:
        dist.recv(t, 0)
elif args.op == "broadcast":
    dist.broadcast(t, 0)
elif args.op == "allreduce":
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
elif args.op == "reduce":
    dist.reduce(t, 0, op=dist.ReduceOp.SUM)
elif args.op == "alltoall":
    dist.all_to_all_single(t2, t)
elif args.op == "alltoallv":
    out_split =[1]*size
    in_split = [1]*size
    dist.all_to_all_single(t2, t, out_split, in_split)

else:
    print("Incorrect operation")
    sys.exit(1)

dist.barrier()
print('rank ', rank, ':', t)
dist.destroy_process_group()
