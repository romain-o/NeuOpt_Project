import time
import torch
import os
from tqdm import tqdm
import random
import numpy as np
from utils.logger import log_to_screen, log_to_tb_val
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from problems.problem_cvrp import CVRPDataset
    
def gather_tensor_and_concat(tensor):
    gather_t = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_t, tensor)
    return torch.cat(gather_t)

def validate(rank, problem, agent, tb_logger, val_dataset=None, distributed = False, _id = None, input_batch = None, conquer=False):
    assert ((not val_dataset == None) or (not input_batch == None)), 'val_dataset or input_batch are both None'
    
    # Validate mode
    opts = agent.opts
    if rank==0: print('\nValidating...', flush=True)
    agent.eval()
    problem.eval()
    if opts.eval_only:
        torch.manual_seed(opts.seed)
        random.seed(opts.seed)
        np.random.seed(opts.seed)
    
    
    if not isinstance(val_dataset, CVRPDataset):
        val_dataset = problem.make_dataset(size=opts.graph_size,
                            num_samples=opts.val_size,
                            filename = val_dataset,
                            DUMMY_RATE = opts.dummy_rate)

    if distributed and opts.distributed:
        device = torch.device("cuda", rank)
        torch.distributed.init_process_group(backend='nccl', world_size=opts.world_size, rank = rank)
        torch.cuda.set_device(rank)
        agent.actor.to(device)
        agent.actor = torch.nn.parallel.DistributedDataParallel(agent.actor, device_ids=[rank])
        if not opts.no_tb and rank == 0:
            tb_logger = SummaryWriter(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, 
                                                          opts.graph_size), opts.run_name))
        assert opts.val_batch_size % opts.world_size == 0
        train_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size = opts.val_batch_size // opts.world_size, shuffle=False,
                                    num_workers=0,
                                    pin_memory=True,
                                    sampler=train_sampler)
    else:
        if val_dataset is not None:
            val_dataloader = DataLoader(val_dataset, batch_size=opts.val_batch_size, shuffle=False,
                                       num_workers=0,
                                       pin_memory=True)
    
    s_time = time.time()
    bv = []
    obj_history = []
    r = []
    
    should_record = opts.record or conquer
    
    def process_batch(batch_):
        rollout_output = agent.rollout(problem=problem,
                                        T=opts.T_max,
                                        val_m=opts.val_m,
                                        stall_limit=opts.stall_limit,
                                        batch=batch_,
                                        record=should_record,
                                        show_bar=rank==0)
        

        if conquer:
            sub_batch = rollout_output[-1]
            res = agent.reconstruct(batch_, sub_batch, rollout_output)
            bv_ = res['total_cost']
            rollout_output = (bv_, rollout_output[1], rollout_output[2])
            
        return rollout_output[:3]
    
    if not input_batch is None:
        print('Processing input batch')
        bv_, obj_history_, r_ = process_batch(input_batch)
        bv.append(bv_)
        obj_history.append(obj_history_)
        r.append(r_)
    else:
        for batch in tqdm(val_dataloader, desc='inference', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):
            bv_, obj_history_, r_ = process_batch(batch)
                    
            bv.append(bv_)
            obj_history.append(obj_history_)
            r.append(r_)
        
    bv = torch.cat(bv, 0)
    obj_history = torch.cat(obj_history, 0)
    r = torch.cat(r, 0)
    
    if distributed and opts.distributed:
        dist.barrier()
        initial_cost = gather_tensor_and_concat(obj_history[:,0,0].contiguous())
        time_used = gather_tensor_and_concat(torch.tensor([time.time() - s_time]).cuda())
        bv = gather_tensor_and_concat(bv.contiguous())
        costs_history = gather_tensor_and_concat(obj_history[:,:,0].contiguous())
        search_history = gather_tensor_and_concat(obj_history[:,:,1].contiguous())
        reward = gather_tensor_and_concat(r.contiguous())
        dist.barrier()
    else:
        initial_cost = obj_history[:,0,0] # bs
        time_used = torch.tensor([time.time() - s_time]) # bs
        bv = bv
        costs_history = obj_history[:,:,0]
        search_history = obj_history[:,:,1]
        reward = r
    
    # log to screen  
    if rank == 0: 
        print(f"\n --- Results ({'CONQUER & RECONSTRUCT' if conquer else 'STANDARD'}) ---")
        if conquer:
            n_splits = opts.dnc_n_splits
            initial_cost = initial_cost.view(-1, n_splits).sum(-1)
            costs_history = costs_history.view(-1, n_splits, opts.T_max + 1).sum(1)
            search_history = search_history.view(-1, n_splits, opts.T_max + 1).sum(1)
            reward = reward.view(-1, n_splits, opts.T_max).sum(1)
            
        log_to_screen(time_used, 
                                initial_cost, 
                                bv, 
                                reward, 
                                costs_history,
                                search_history,
                                batch_size = opts.val_size, 
                                T = opts.T_max)
  
    
    # log to tb
    if(not opts.no_tb) and rank == 0:
        log_to_tb_val(tb_logger,
                      time_used, 
                      initial_cost, 
                      bv, 
                      reward, 
                      costs_history,
                      search_history,
                      val_size = opts.val_size,
                      T = opts.T_max,
                      epoch = _id)
    