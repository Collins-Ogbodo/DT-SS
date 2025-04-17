import numpy as np
import gymnasium
import random
import logging
from gymnasium import spaces
from typing import Optional, Tuple
from scipy.linalg import solve_triangular
from sklearn.preprocessing import OneHotEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CantileverEnv_v0_1(gymnasium.Env):
    """
    A reinforcement learning environment for optimal sensor placement on a cantilever beam using OpenAI Gym.
    
    Spaces:
        - Action Space: MultiDiscrete (sensor_id, direction)
        - Observation Space: Box for the binary representation of sensor states.
        - Direction ACTION = { "LEFT" : 0,
                  "RIGHT" : 1,
                  'UP'    : 2,
                  'Down': 3}
    """
    ACTIONS = ["LEFT", "RIGHT", "UP", "DOWN"]
    
    def __init__(self, config: dict) -> None:
        super().__init__()
        #Initialisation
        self.num_sensors = config.get("num_sensors", 4)
        self.seed = config.get("seed", 0)
        random.seed(self.seed)
        self.render = config.get("render",False)
        self.episode_length = config.get("episode_length",1000)
        # Set up action and observation spaces
        self.action_space = spaces.MultiDiscrete([self.num_sensors, len(self.ACTIONS)], seed=self.seed)
        #Number of candidate location
        num_nodes = 1462
        self.num_conditions = config.get("num_conditions", 0) + 1 #+1 for the healthy case
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_nodes + self.num_conditions,), dtype=np.int8, seed=self.seed)

        #Condition encoding
        encoder = OneHotEncoder(sparse_output=False)
        self.enc_cond = encoder.fit_transform(np.arange(self.num_conditions).reshape(-1, 1))
        
        # Initialize PyAnsys environment
        self.pyansys_env = config.get('pyansys_env')
        #precompute mode shape and spatial correlation matrices
        self.phi_list = self.pyansys_env.phi_list
        self.correlation_covariance_matrix_list = self.pyansys_env.correlation_covariance_matrix_list       
                
        #Direction Adjustments
        self.adjustments = [
            [0.0, 0.0, -4.7625e-3],  # LEFT
            [0.0, 0.0, 4.7625e-3],   # RIGHT
            [4.9765e-3, 0.0, 0.0],   # UP
            [-4.9765e-3, 0.0, 0.0]   # DOWN
        ]
        
        #Heuristic Score for normalisation : Human score
        if config.get("condition_case") == 'severity':
            self.norm_factor = np.array([7.3266, 0.7617, 1.8836]) #[Health, Condtion 1, condtion 2] sensors Damage severity
        elif config.get("condition_case") == 'localisation': 
            self.norm_factor = np.array([12.8112, 4.0689, 4.0504]) #[Health, Condtion 1, condtion 2] sensors Damage location
            

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Takes an action and returns the next state, reward, and termination status.
        
        Args:
            action (np.ndarray): Action array in the form [sensor_id, direction].

        Returns:
            the new observation, reward, termination status, and info dict.
        """
        try:
            self.step_left -= 1
            self.current_state = self._action_state_map(self.current_state, action)
            self.pyansys_env.render(self.current_state) if self.render == True else None
            self.current_state_binary = np.isin(self.pyansys_env.observation_space_node, self.current_state).astype(np.int8)
            reward = self._get_reward(self.current_state)
            done = self._is_done(self.current_state, action)
            info = {'node_Id': self.current_state, 'reward_metric': self.reward_metric_new}
            return np.concatenate((self.current_state_binary, self.bin_encode)), reward, done, False, info
        
        except Exception as e:
            logging.error(f"Error during step execution: {e}")
            raise

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment and returns the initial state.
        """
        try:
            #Initialise epsiode length and state
            self.step_left = self.episode_length
            """condition randomisation
                Josh Tobin, Rachel Fong, Alex Ray, Jonas Schneider, Wojciech Zaremba, and Pieter Abbeel. Domain randomiza
                tion for transferring deep neural networks from simulation to the real world. In 2017 IEEE/RSJ international
                conference on intelligent robots and systems (IROS), pages 23–30. IEEE, 2017.
            """
            self.phi_index = np.random.choice(self.num_conditions, replace= True)
            self.phi = self.phi_list[self.phi_index]
            self.correlation_covariance_matrix = self.correlation_covariance_matrix_list[self.phi_index]
            #Binary encoding
            self.bin_encode = self.enc_cond[self.phi_index, :]
            #State initilisation
            self.current_state = self._initialize_state()
            #render initialise state
            self.pyansys_env.render(self.current_state) if self.render == True else None
            #Binary encoding of inititalised state
            self.current_state_binary = np.isin(self.pyansys_env.observation_space_node, self.current_state).astype(np.int8)
            #compute reward metric
            reward_metric_start = self._info_metric(self.current_state)
            #recursive reward
            self.reward_metric_old = reward_metric_start         
            #Info logging
            info = {'node_Id':self.current_state, 'reward_metric': reward_metric_start}            
            return np.concatenate((self.current_state_binary, self.bin_encode)), info
        
        except Exception as e:
            logging.error(f"Error during environment reset: {e}")
            raise

    def _initialize_state(self):
        """
        Heuristic Initialisation of the starting state for the sensors.
        """
        # Calculate indices of equally spaced nodes
        all_init_state = np.array([2356, 2347, 2339, 2332, 2327, 2321, 2314, 2307, 2301, 2295, 2288, 2279])
        indices = np.linspace(0, len(all_init_state) - 1, self.num_sensors, dtype=int)
        init_state = all_init_state[indices]
        return np.array(init_state)

    def _is_done(self, state: np.ndarray, action: np.ndarray) -> bool:
        """
        Checks if the episode is complete based on step count.
        """
        return self.step_left <= 0 


    def _get_reward(self, state: np.ndarray) -> float:
        """
        Calculates the reward for the current state based on Fisher Information matrix.
        """
        # Check if the state contains unique sensor placements
        if len(np.unique(state)) != self.num_sensors:
            return -1  # Penalize invalid sensor configurations
        try:
            self.reward_metric_new = self._info_metric(state)
            reward = self.reward_metric_new - self.reward_metric_old
            self.reward_metric_old = self.reward_metric_new
            return reward
        except ValueError as e:
            logging.warning(f"Reward calculation error: {e}")
            raise


    def _info_metric(self, state: np.ndarray) -> float:
        """
        Calculates the information entropy index for the sensor placements.
        
        The reward function is an information theoritic metric which measures the information gain for each configuration
            References: 
            [1] Zhang, Jie, et al. "Optimal sensor placement for multi-setup modal analysis of structures." 
                Journal of Sound and Vibration 401 (2017): 214-232.
            [2] Papadimitriou, Costas, and Geert Lombaert. "The effect of prediction error correlation on optimal sensor placement in structural dynamics." 
                Mechanical Systems and Signal Processing 28 (2012): 105-127.
            [3] Wang, Ying, et al. "Advancements in Optimal Sensor Placement for Enhanced Structural Health Monitoring: Current Insights and Future Prospects." 
                Buildings 13.12 (2023): 3129.
            [4] Wang, Zhi, Han-Xiong Li, and Chunlin Chen. "Reinforcement learning-based optimal sensor placement for spatiotemporal modeling." 
                IEEE transactions on cybernetics 50.6 (2019): 2861-2871.
            [5] Kammer, Daniel C. "Sensor placement for on-orbit modal identification and correlation of large space structures." 
                Journal of Guidance, Control, and Dynamics 14.2 (1991): 251-259.
            [6] Tcherniak, Dmitri. "Optimal Sensor Placement: a sensor density approach." (2022).
            [7] Papadimitriou, Costas. "Optimal sensor placement methodology for parametric identification of structural systems." 
                Journal of sound and vibration 278.4-5 (2004): 923-947.
            [8] Papadimitriou, Costas, James L. Beck, and Siu-Kui Au. "Entropy-based optimal sensor location for structural model updating." 
                Journal of Vibration and Control 6.5 (2000): 781-800.
            [9] Papadimitriou, Costas. "Optimal sensor placement methodology for parametric identification of structural systems." 
                Journal of sound and vibration 278.4-5 (2004): 923-947.
        """
        try:
            # Vectorized construction of L_mat
            L_mat = (self.pyansys_env.observation_space_node[:, None] == state).astype(np.int8).T
            #Evaluate FIM using Cholesky decomposition
            sigma_C = np.linalg.cholesky(L_mat @ self.correlation_covariance_matrix @ L_mat.T)
            sigma_A = solve_triangular(sigma_C, (L_mat @ self.phi), lower= True)
            sing_A = np.linalg.svd(sigma_A, compute_uv= False)
            return np.prod(np.square(sing_A))/self.norm_factor[self.phi_index]
        except np.linalg.LinAlgError:
            logging.error("Error in Fisher information matrix computation.")
            raise

    def _action_state_map(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Maps action to new state coordinates.
        """
        try:
            sensor_id, direction = action
            state_node_id = state[sensor_id]
            # Extract the coordinates of the current state node
            state_row = self.pyansys_env.coord_2d_array[self.pyansys_env.coord_2d_array[:, 0] == state_node_id, 1:][0]
            # Apply the action adjustment
            new_state_coord = self._apply_action_adjustment(state_row, direction)
            # Compute distances and find the nearest node
            distances = np.linalg.norm(self.pyansys_env.coord_2d_array[:, 1:] - new_state_coord, axis=1)
            state[sensor_id] = self.pyansys_env.coord_2d_array[np.argmin(distances), 0]
            return state
        except IndexError:
            logging.warning("Invalid state transition detected.")
            raise

    def _apply_action_adjustment(self, state_row, direction):
        return np.round(state_row + self.adjustments[direction], 4)

    def close(self) -> None:
        self.pyansys_env.close()