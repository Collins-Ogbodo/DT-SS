import os
import random
import numpy as np
import torch
import datetime
from tianshou.data import (
    Collector,
    PrioritizedVectorReplayBuffer,
    VectorReplayBuffer,
)
from tianshou.policy import RainbowPolicy
from tianshou.policy.base import BasePolicy
from tianshou.policy.modelfree.rainbow import RainbowTrainingStats
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net
from tianshou.utils import WandbLogger
from torch.utils.tensorboard import SummaryWriter 
from tianshou.env import DummyVectorEnv
from env.utils import FlattenMultiDiscreteActions
from env.cantilever_env import CantileverEnv_v0_1
from env.pyansys_sim import Cantilever

config_kwargs = {
    "task" : "CantileverEnv_v0_1-Wrapped",
    "batch_size": 32,
    "gamma" : 0.9,
    "seed"  : 0,
    "eps_train_initial"  : 1.0, 
    "eps_train_final"  : 0.01, 
    "eps_test"  : 0.0,
    "buffer_size"  : 1_000_000,
    "num_atoms" : 51, 
    "v_min"  : -10.0,
    "v_max" : 10.0,
    "step_per_collect"  : 4,
    "hidden_sizes" : [256, 128],                                       #[128, 128]
    "num_train_env"  : 2,
    "num_test_env"  : 2,
    "logdir"  : "cantilever-0-1-2-damages-Mode-1-2-3-3_Sensors",
    "prioritized_replay" : True,
    "beta"  : 0.4,
    "beta_final"  : 1.0,
    "device"  : "cpu", 
    'wandb_project': "severity-cantilever-0-1-2-damages-Mode-1-2-3-3_Sensors",
    'model_name': "severity-cantilever-0-1-2-damages-Mode-1-2-3-3_Sensors",
    'replay_buffer_name': "buffer-PC",
    "epoch"  : 100,                                                    # 200 
    'episode_per_test' : 3,
    "step_per_epoch"  : 10_000,
    "decay_steps" : 250_000,
    "target_network_update_freq" : 3200,
    'verbose_ax': False,
    "sim_modes": [0,1,2],
    "num_sensors": 3,                                                #4
    "num_conditions" : 2,
    "render" : False,
    "norm" : True,
    "episode_length" : 1000,
    "learning_rate_denom" : 6,                                       #2
    "priority_exponent" : 0.5,                                       #0.7
    "multi_step_returns" : 3,                                        #5
    "condition_case" : 'severity',                                   #'localisation'
    "node_id" : [2983,3019],                                         #[90, 1670]
    "mass" : [0.7, 0.7]                                              #[0.2, 0.2] kg
}


def set_random_seeds(seed: int, using_cuda: bool = False) -> None:
  """
  Seed the different random generators.
  """
  # Set seed for Python random, NumPy, and Torch
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

  # Set deterministic operations for CUDA
  if using_cuda:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_rainbow(config_kwargs) -> float:
    #Environment setup
    pyansys_env = Cantilever(config_kwargs)
    config_kwargs['pyansys_env'] = pyansys_env

    train_envs = DummyVectorEnv([lambda: FlattenMultiDiscreteActions(CantileverEnv_v0_1(config_kwargs)) \
                                 for _ in range(config_kwargs.get("num_train_env"))])
    test_envs = DummyVectorEnv([lambda: FlattenMultiDiscreteActions(CantileverEnv_v0_1(config_kwargs)) \
                                for _ in range(config_kwargs.get("num_test_env"))])
    state_shape = train_envs.observation_space[0].shape
    action_shape = train_envs.action_space[0].n
    
    # Set random seed
    set_random_seeds(config_kwargs["seed"], using_cuda=torch.cuda.is_available())
        
    Q_param = {"hidden_sizes": config_kwargs.get("hidden_sizes")}
    V_param = {"hidden_sizes": config_kwargs.get("hidden_sizes")}
    net = Net(
        state_shape= state_shape,
        action_shape= action_shape,
        hidden_sizes= config_kwargs.get("hidden_sizes"),
        device= config_kwargs.get("device"),
        softmax=True,
        num_atoms= config_kwargs.get("num_atoms"),
        dueling_param= (Q_param, V_param))
    optim = torch.optim.Adam(net.parameters(), 
                             lr= 0.00025 / config_kwargs['learning_rate_denom'], eps = 1.5e-4)
    policy: RainbowPolicy[RainbowTrainingStats] = RainbowPolicy(
        model=net,
        optim=optim,
        discount_factor= config_kwargs['gamma'],
        action_space=train_envs.action_space[0],
        num_atoms= config_kwargs.get("num_atoms"),
        v_min= config_kwargs.get("v_min"),
        v_max= config_kwargs.get("v_max"),
        estimation_step=  config_kwargs['multi_step_returns'],
        target_update_freq= config_kwargs['target_network_update_freq'],
    ).to( config_kwargs.get("device"))
    # buffer
    buf_train: PrioritizedVectorReplayBuffer | VectorReplayBuffer
    if  config_kwargs.get("prioritized_replay"):
        buf_train = PrioritizedVectorReplayBuffer(
             config_kwargs.get("buffer_size"),
            buffer_num=config_kwargs.get("num_train_env"),
            alpha= config_kwargs['priority_exponent'],
            beta= config_kwargs.get("beta"),
            weight_norm=True,
        )
    else:
        buf_train = VectorReplayBuffer( config_kwargs.get("buffer_size"), buffer_num= config_kwargs.get("num_train_env"))
        
    
    buf_test = VectorReplayBuffer(config_kwargs.get("num_test_env") *config_kwargs.get('episode_length') 
                                 *config_kwargs.get("episode_per_test"), buffer_num= config_kwargs.get("num_test_env")) 
    # collector
    train_collector = Collector(policy, train_envs, buf_train, exploration_noise=True)
    test_collector = Collector(policy, test_envs, buf_test,exploration_noise=True)
 
    train_collector.reset()
    train_collector.collect(n_step= config_kwargs['batch_size'] * config_kwargs.get("num_train_env"))
    
    # log time
    dt = datetime.datetime.now(datetime.timezone.utc)
    dt = dt.replace(microsecond=0, tzinfo=None)
    # logger    
    wandb_logger = WandbLogger(project= config_kwargs.get("wandb_project"),
                         name= str(dt),
                         config = config_kwargs)

    log_path = os.path.join( config_kwargs.get("logdir"),  config_kwargs.get("task"), "rainbow")
    if not os.path.exists(log_path): os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    wandb_logger.load(writer)
    

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "best_policy.pth"))

    total_steps = config_kwargs.get("epoch") * config_kwargs.get("step_per_epoch")
    
    def train_fn(epoch: int, env_step: int) -> None:
        
        if env_step <= config_kwargs.get("decay_steps"):
            eps = config_kwargs.get("eps_train_initial") - (env_step / config_kwargs.get("decay_steps")) *(
            config_kwargs.get("eps_train_initial") - config_kwargs.get("eps_train_final"))
        else:
            eps = 0.01
        policy.set_eps(eps)
        # beta annealing, as discribed in the paper
        # Linearly increase beta from 0.4 to 1
        beta = config_kwargs.get("beta") + ((config_kwargs.get("beta_final") - config_kwargs.get("beta")) * env_step / total_steps)
        # Set beta in your buffer
        buf_train.set_beta(beta)  
     
    #Get final state reward metric
    def find_last_non_zero(lst): 
        arr = np.array(lst) 
        non_zero_indices = np.nonzero(arr)[0] 
        if len(non_zero_indices) == 0: 
            return None 
        return arr[non_zero_indices[-1]]
    
    def info_data(env_step):       
        #Extract reward metric from test buffer           
        rew_metric = buf_test.get(np.arange(config_kwargs.get("num_test_env") *config_kwargs.get('episode_length') 
                                            * config_kwargs.get("episode_per_test")),"info" )['reward_metric']
        list_test_reward_metric = np.array_split(rew_metric, config_kwargs.get("episode_per_test"))

        avg_ep_rew_metric_final = np.mean([ find_last_non_zero(ep_rew_metric) for ep_rew_metric in list_test_reward_metric])
        print([ find_last_non_zero(ep_rew_metric) for ep_rew_metric in list_test_reward_metric])
        avg_ep_rew_metric_sum = np.mean([ np.sum(ep_rew_metric) for ep_rew_metric in list_test_reward_metric])
        print([ np.sum(ep_rew_metric) for ep_rew_metric in list_test_reward_metric])
        wandb_logger.write('test/env_step', env_step, {'avg_ep_rew_metric_sum': avg_ep_rew_metric_sum})
        wandb_logger.write('test/env_step', env_step, {'avg_ep_rew_metric_final': avg_ep_rew_metric_final})
        buf_test.reset()  
    
    def test_fn(epoch: int, env_step: int | None) -> None:
        policy.set_eps( config_kwargs.get("eps_test"))
        if epoch >= 2:
           #log data manually
           info_data(env_step)

    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch= config_kwargs.get("epoch"),
        step_per_epoch= config_kwargs.get("step_per_epoch"),
        step_per_collect= config_kwargs.get("step_per_collect"),
        episode_per_test= config_kwargs.get("episode_per_test"),
        batch_size= config_kwargs['batch_size'],
        update_per_step= 1/ config_kwargs.get("step_per_collect"),
        train_fn=train_fn,
        logger= wandb_logger,
        test_fn=test_fn,
        save_best_fn=save_best_fn
    ).run()
    wandb_logger.finalize()
    #Stats for last test         
    rew_metric = buf_test.get(np.arange(config_kwargs.get("num_test_env") *config_kwargs.get('episode_length') 
                                        * config_kwargs.get("episode_per_test")),"info" )['reward_metric']
    list_test_reward_metric = np.array_split(rew_metric, config_kwargs.get("episode_per_test"))

    avg_ep_rew_metric_final = np.mean([ find_last_non_zero(ep_rew_metric) for ep_rew_metric in list_test_reward_metric])
    print([ find_last_non_zero(ep_rew_metric) for ep_rew_metric in list_test_reward_metric])
    avg_ep_rew_metric_sum = np.mean([ np.sum(ep_rew_metric) for ep_rew_metric in list_test_reward_metric])
    final_node_id = buf_test.get(np.arange(config_kwargs.get("num_test_env") *config_kwargs.get('episode_length') 
                                        * config_kwargs.get("episode_per_test")),"info" )['node_Id'][-1]
    torch.save(policy.state_dict(), os.path.join(log_path, "final_policy.pth"))
    return result.best_reward, policy, test_collector, avg_ep_rew_metric_final, avg_ep_rew_metric_sum, final_node_id

# Perform trial
best_reward, policy, test_collector, avg_ep_rew_metric_final, avg_ep_rew_metric_sum, final_node_id = test_rainbow(config_kwargs)

last_rew_metric_data = {"avg_ep_rew_metric_final": avg_ep_rew_metric_final,
                        "avg_ep_rew_metric_sum": avg_ep_rew_metric_sum,
                        "final_node_id": final_node_id}
#Log environment parameters
with open(os.path.join(config_kwargs["logdir"] , 'Config_file.txt'), 'w') as txt_file:
    for key, value in config_kwargs.items():
        txt_file.write(f'{key}: {value}\n')
    for key, value in last_rew_metric_data.items():
        txt_file.write(f'{key}: {value}\n') 
