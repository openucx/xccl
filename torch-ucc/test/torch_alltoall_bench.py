import argparse
import torch
import torch.distributed as dist
import sys
import os
from time import perf_counter
import torch_ucc

def get_tensor(size, device, val):
    count = size//4
    t = torch.ones([count], dtype=torch.int32, device=device)
    t = t + val
    return t

parser = argparse.ArgumentParser(description="Process Group Alltoall Benchmark")
parser.add_argument("--backend", type=str, default='mpi')
parser.add_argument("--use-cuda", default=False, action='store_true')
parser.add_argument("--min-size", type=int, default=2**5)
parser.add_argument("--max-size", type=int, default=2**15)
parser.add_argument("--skip", type=int, default=500)
parser.add_argument("--iter", type=int, default=100)
args = parser.parse_args()

try:
    comm_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    comm_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
except:
    print('OMPI env variables are not found')
    sys.exit(1)

os.environ['MASTER_PORT'] = '32167'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['RANK']        = str(comm_rank)
os.environ['WORLD_SIZE']  = str(comm_size)

if args.use_cuda and not torch.cuda.is_available():
    print("CUDA is not available")
    sys.exit(0)

if args.backend=='nccl' and not args.use_cuda:
    print("NCCL backend doesn't support host buffers")
    sys.exit(0)

if args.use_cuda:
    torch.cuda.set_device(comm_rank)
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

if comm_rank == 0:
    print("World size {}".format(comm_size))
    print("%-10s %-10s %-10s %-10s" %('size', 'min, us', 'avg, us', 'max, us'))

if args.backend != 'mpi':
    dist.init_process_group(args.backend, rank=comm_rank, world_size=comm_size)
else:
    dist.init_process_group(args.backend)

size = args.min_size
while size <= args.max_size:
    bufsize = size * comm_size
    send_tensor = get_tensor(bufsize, args.device, comm_rank)
    recv_tensor = get_tensor(bufsize, args.device, 0)
    time = 0
    for i in range(args.iter + args.skip):
        start = perf_counter()
        req = dist.all_to_all_single(recv_tensor, send_tensor, async_op=True)
        #req = dist.all_reduce(send_tensor, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()
        torch.cuda.synchronize(args.device)
        finish = perf_counter()
        dist.barrier()
        if  i > args.skip:
            time += finish - start
    time = [time / args.iter]
    max_time = torch.tensor([time], device=args.device)
    min_time = torch.tensor([time], device=args.device)
    avg_time = torch.tensor([time], device=args.device)
    dist.all_reduce(max_time, op=dist.ReduceOp.MAX)
    dist.all_reduce(min_time, op=dist.ReduceOp.MIN)
    dist.all_reduce(avg_time, op=dist.ReduceOp.SUM)
    if comm_rank == 0:
        print("%-10i %-10.3f %-10.3f %-10.3f" %(size, min_time[0] * (10**6), avg_time[0] * (10**6)/comm_size, max_time[0] * (10**6)))
    size = size * 2
    