import numpy as np
import gymnasium as gym
import pandas as pd
import pyuff
import re

class FlattenMultiDiscreteActions(gym.ActionWrapper):
    """
    A reinforcement learning environment for optimal sensor placement on a cantilever beam using OpenAI Gym.
    
    Spaces:
        - Action Space: Discrete (number_of_sensor * direction)
        - Observation Space: Box for the binary representation of sensor states.
        - Direction ACTION = { "LEFT" : 0,
                  "RIGHT" : 1,
                  'UP'    : 2,
                  'Down': 3}
        -action = {Sensor1 and Left : 0,
                   Sensor1 and Right: 1,
                   Sensor1 and Up : 2,
                   Sensor1 and Down : 3,
                   Sensor2 and Left : 4,
                   Sensor2 and Right: 5,
                   Sensor2 and Up : 6,
                   Sensor2 and Down : 7,
                   ...
                   SensorN and all combination of direction: N *4}
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.action_space, gym.spaces.MultiDiscrete)
        self.nvec = self.action_space.nvec
        self.action_space = gym.spaces.Discrete(np.prod(self.action_space.nvec))

    def action(self, action):
        actions = []
        actions.append(action//self.nvec[-1])
        actions.append(action%self.nvec[-1])
        return np.array(actions)

def effective_independence(mode_shape, num_sensors, spatial_cov):
    """
    Perform Effective Independence (EfI) sensor placement.
    
    Parameters:
    mode shape matric (numpy.ndarray).
    num_sensors (int): Number of sensors to place.
    spatial_cov(numpy.ndarray): spatial correclation matrix
    
    Returns:
    numpy.ndarray: Indices of the selected sensor locations.
    """
    #Compute FIM
    FIM = mode_shape.T @ mode_shape
    # Initialize sensor indices
    sensor_indices = np.arange(mode_shape.shape[0])
    spatial_cov_inv = np.linalg.inv(spatial_cov)
    # Calculate the initial EfI values
    efi_vector = np.diag(spatial_cov_inv @ mode_shape @ np.linalg.inv(FIM) @ mode_shape.T @ spatial_cov_inv)
    efi_vector = np.multiply(efi_vector, np.diag(spatial_cov_inv))
    while len(sensor_indices) != num_sensors:
        # Select the sensor with the lowest EfI value
        min_index = np.argmin(efi_vector)

        # Remove the selected sensor from the list
        sensor_indices = np.delete(sensor_indices, (min_index))

        # Update the Mode shape matrix by removing the selected sensor
        mode_shape = np.delete(mode_shape, min_index, axis=0)
        spatial_cov_inv = np.delete(spatial_cov_inv, min_index, axis=0)
        spatial_cov_inv = np.delete(spatial_cov_inv, min_index, axis=1)
        # Recalculate the EfI values
        FIM = mode_shape.T @ mode_shape
        efi_vector = np.diag(spatial_cov_inv @ mode_shape @ np.linalg.inv(FIM) @ mode_shape.T @ spatial_cov_inv)
        efi_vector = np.multiply(efi_vector, np.diag(spatial_cov_inv))
    return sensor_indices, efi_vector


def MAC(mode_shape_sim : np.array, 
        mode_shape_mea: np.array,
        wn_exp,
        wn_ana) -> np.array:
    """
    Evaluate MAC.
    Parameters:
    -----------
    mode_shape_sim (list): mode shape values from model (simulated)
    mode_shape_mea (list): mode shape values from experiment (EMA)
    Return:
    --------
    function (np.array): MAC
    """
    mode_length, _num_sensors = mode_shape_sim.shape 
    mode_shape_sim = mode_shape_sim.reshape(mode_length, _num_sensors)
    mode_shape_mea = mode_shape_mea.reshape(mode_length, _num_sensors)
    MAC = np.zeros((mode_length, mode_length))  
    for i, sim in enumerate(mode_shape_sim):
            for j, mea in enumerate(mode_shape_mea):
                num = np.real(np.abs(mea.T @ np.conj(sim))**2)
                den = np.real((mea.T @ np.conj(mea)) * (sim.T @ np.conj(sim)))
                MAC[i, j] = num / den  # Store the MAC value in the i,j position
    mac_df = pd.DataFrame(MAC, index=np.round(wn_exp,2), columns= np.round(wn_ana,2))
    return mac_df

def parse_unv_file(file_path: str, EMA_modes: list) -> tuple:
    """
    Extract experimental modal parameters.
    Parameters:
    -----------
    file_path (str): path to universal file with modal parameters
    Return:
    --------
    wn (list): list of natural frequencies.
    mode_shape (list of list): List of model shapes for all modes
    """
    #Extract EMA mode shape and Natural frequency
    mea_data = pyuff.UFF(file_path).read_sets()
    wn = []
    #dam = []
    mode_shape = []
    EMA_modes = np.array(EMA_modes)
    for dic in mea_data:
        try:
            if dic['type'] == 55: #and dic['id1'] == 'PolyMAX-Ansys-Up': #Ensure you change this when you get new EMA data
                # Split the string by commas
                parts = dic['id4'].split(',')
                # Extract the numbers using regular expressions
                #mode_no = re.search(r'MODE NO\. (\d+)', parts[0]).group(1)
                wn.append(re.search(r'FREQUENCY ([\d\.]+)\(Hz\)', parts[1]).group(1))
                #dam.append(re.search(r'DAMPING ([\d\.]+)', parts[2]).group(1))
                mode_shape.append(dic['r3']) #you have to come back and generalise this so it can cover for r1, r2, r3 depending on where the data is saved
        except:
            print('parsing error')
    return (np.array(wn, dtype = float)[EMA_modes], np.array(mode_shape)[EMA_modes[:, np.newaxis], :])   