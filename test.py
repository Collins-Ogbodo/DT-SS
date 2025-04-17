import os
import random
import numpy as np
import torch
from tianshou.data import (
    Collector,
    VectorReplayBuffer,
)
from tianshou.policy import RainbowPolicy
from tianshou.policy.modelfree.rainbow import RainbowTrainingStats
from tianshou.utils.net.common import Net
from env.utils import FlattenMultiDiscreteActions, effective_independence, MAC, parse_unv_file
from env.cantilever_env import CantileverEnv_v0_1
from env.pyansys_sim import Cantilever

config_kwargs = {
    "task" : "CantileverEnv_v0_1-Wrapped",
    "gamma" : 0.9,
    "seed"  : 0,
    "num_atoms" : 51, 
    "v_min"  : -10.0,
    "v_max" : 10.0,
    "hidden_sizes" : [256, 128],                                     #[128, 128]
    "num_test_env"  : 2,
    "logdir"  : "cantilever-0-1-2-damages-Mode-1-2-3-3_Sensors",
    "project_name" : "rainbow",
    "device"  : "cpu",                                               # 200 
    'episode_per_test' : 3,
    "target_network_update_freq" : 3200,
    "sim_modes": [0,1,2],
    "num_sensors": 3,                                                #4
    "num_conditions" : 2,
    "render" : True,
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


#Environment setup
pyansys_env = Cantilever(config_kwargs)
config_kwargs['pyansys_env'] = pyansys_env

envs = FlattenMultiDiscreteActions(CantileverEnv_v0_1(config_kwargs))
state_shape = envs.observation_space.shape
action_shape = envs.action_space.n

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
    action_space=envs.action_space,
    num_atoms= config_kwargs.get("num_atoms"),
    v_min= config_kwargs.get("v_min"),
    v_max= config_kwargs.get("v_max"),
    estimation_step=  config_kwargs['multi_step_returns'],
    target_update_freq= config_kwargs['target_network_update_freq'],
).to( config_kwargs.get("device"))


log_path = os.path.join( config_kwargs.get("logdir"),  config_kwargs.get("task"), config_kwargs.get("project_name"))
#policy.load_state_dict(torch.load(os.path.join(log_path,"final_policy.pth")))
policy.load_state_dict(torch.load(os.path.join(log_path, "best_policy.pth")))
policy.eval()
buf_test = VectorReplayBuffer(config_kwargs.get("num_test_env") *config_kwargs.get('episode_length') \
                              *config_kwargs.get("episode_per_test"), buffer_num= config_kwargs.get("num_test_env")) 
buf_test.reset()
eval_collector = Collector(policy, envs, buf_test, exploration_noise=False)
eval_collector.collect(n_episode = 1, reset_before_collect=True)
# Save evaluation results
ep_rew = np.sum(buf_test.get(np.arange(config_kwargs.get('episode_length')),"rew" ))
rew_metric = buf_test.get(np.arange(config_kwargs.get('episode_length')),"info" )['reward_metric']
ep_rew_metric_sum  = np.sum(rew_metric)
ep_rew_metric_final = rew_metric[-1]
final_node_id = buf_test.get(np.arange(config_kwargs.get('episode_length')),"info" )['node_Id'][-1]

print('----------------------------------------------------------------')
print(f"Evaluation Final node_id: {final_node_id}")
print(f"Evaluation episode reward: {ep_rew}")
print(f"Evaluation episode reward metric sum: {ep_rew_metric_sum}")
print(f"Evaluation episode reward metric final: {ep_rew_metric_final}")

"""Effective Independence
"""

mode_shape = envs.phi_list[2]
spatial_cov = envs.correlation_covariance_matrix_list[2]
sensor_indices, efi_vector = effective_independence(mode_shape, config_kwargs.get('num_sensors'), spatial_cov)

observation_space_node = envs.pyansys_env.observation_space_node[[sensor_indices]]
print("final node:", observation_space_node )

"""Experimental Modal Analysis (EMA) data

Node - Sensor location Matching
----------------------
|   EMA      |  FEM  |
----------------------
|    1       | 2960  |
----------------------
|    3       | 18    |
----------------------
|    4       | 2974  |
----------------------
|    6       | 32    |
----------------------
|    7       | 2990  |
----------------------
|    9       | 48    |
----------------------
|    10       | 3002 |
----------------------
|    12       | 60   |
----------------------
|    13       | 3017 |
----------------------
|    15       | 75   |
----------------------
|    16       | 1670 |
----------------------
|    18       | 90   |
----------------------
"""
UNV_path = os.path.join(os.getcwd(),"env","EMA","512Hz-Burst90%-12Sensors-Polymax-Complex.unv")
mea_node = np.array([2960, 2990, 60, 1670, 3002, 18, 48, 32, 75, 90, 3017, 2974])
EMA_modes = [0, 3, 5, 9, 15, 16]
"""Modal Assurance Criterion - MAC
"""
num_modes = len(config_kwargs["sim_modes"])
wn_exp, phi_exp = parse_unv_file(UNV_path, EMA_modes[:num_modes])
phi_exp = phi_exp.reshape(num_modes, len(mea_node))
phi_ana = pyansys_env.extract_mode_shape(mea_node)
wn_ana = pyansys_env.wn
mac_mat = MAC(phi_ana, phi_exp, wn_exp, wn_ana)
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
sns.heatmap(mac_mat, annot=True, fmt=".2f", cmap='YlGnBu')
plt.title('MAC Matrix')
plt.xlabel('Analytical Natural Frequencies')
plt.ylabel('Physical Structure Natural Frequencies')
plt.show()
