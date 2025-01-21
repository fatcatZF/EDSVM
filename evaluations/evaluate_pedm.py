import os 
import glob 
import json 

import numpy as np 

from sklearn.metrics import roc_auc_score

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from torchensemble import BaggingRegressor
from torchensemble.utils import io 

from gymnasium.spaces.discrete import Discrete
from gymnasium.spaces.box import Box

from stable_baselines3 import DQN, PPO, SAC 

from environment_util import make_env 

import argparse 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="cartpole", help="name of environment")
    parser.add_argument("--policy-type", type=str, default="dqn", help="type of rl policy")
    parser.add_argument("--env0-steps", type=int, default=1000, help="Validation Steps")
    parser.add_argument("--env1-steps", type=int, default=3000, help="Undrifted Steps")
    parser.add_argument("--env2-steps", type=int, default=3000, help="Semantic Drift Steps")
    parser.add_argument("--env3-steps", type=int, default=3000, help="Noisy Drift Steps")
    parser.add_argument("--n-exp-per-model", type=int, default=10, 
                        help="number of experiments of each trained model.") 

    args = parser.parse_args() 

    allowed_envs = {"cartpole", "lunarlander", "hopper", 
                    "halfcheetah", "humanoid"}
    
    allowed_policy_types = {"dqn", "ppo", "sac"}
    
    if args.env not in allowed_envs:
        raise NotImplementedError(f"The environment {args.env} is not supported.")
    if args.policy_type not in allowed_policy_types:
        raise NotImplementedError(f"The policy {args.policy_type} is not supported.")

    print("Parsed arguments: ")
    print(args) 

    # Load trained Agent
    if (args.policy_type=="dqn"):
        AGENT = DQN 
    elif (args.policy_type=="ppo"):
        AGENT = PPO 
    else:
        AGENT = SAC 

    
    policy_env_name = args.policy_type + '-' + args.env

    agent_path = os.path.join('../agents/', policy_env_name)
    agent = AGENT.load(agent_path) 
    print("Successfully Load Trained Agent.")

    ## Define PEDM Base
    ### Compute input dim
    obs_dim = agent.observation_space.shape[-1]
    if isinstance(agent.action_space, Box):
       action_dim = agent.action_space.shape[-1]
    else:
       action_dim = 1
    class BaseNN(nn.Module):
      def __init__(self, input_dim=obs_dim+action_dim, hidden_dim=500, 
                   output_dim=obs_dim):
        super(BaseNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, output_dim)
        self.fc_var = nn.Linear(hidden_dim, output_dim)

      def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        var = torch.exp(log_var)
        output = torch.cat([mu, var], dim=-1)
        return output
      
    
    # Load Drift Detector Models
    loaded_models = []
    model_folder = os.path.join("../models/PEDM", policy_env_name)
    model_pattern = os.path.join(model_folder,  f"pedm_*")
    matching_models = glob.glob(model_pattern)  

    print(matching_models) 
    if len(matching_models)==0:
        raise NotImplementedError(f"There is no trained PEDM for the environment {args.env}.")
    
    for model_path in matching_models:
       pedm = BaggingRegressor(
          estimator = BaseNN,
          n_estimators = 5,
          cuda = False,
       )
       io.load(pedm, model_path)
       loaded_models.append(pedm)

    print(f"Number of trained models: {len(loaded_models)}")



if __name__ == "__main__":


    class MultivariateGaussianNLLLossCustom(nn.Module):
      def __init__(self, reduction='mean'):
        super(MultivariateGaussianNLLLossCustom, self).__init__()
        self.reduction = reduction

      def forward(self, output, y):
        # Ensure variance is positive for each dimension
        dim = output.shape[-1] // 2
        mu, var = output[:, :dim], output[:, dim:]
        var = torch.clamp(var, min=1e-6)  # To avoid division by zero or log of zero

        # Compute the negative log-likelihood for each dimension and sum across dimensions
        nll = 0.5 * ((y - mu) ** 2 / var + torch.log(var))

        # Sum across the output dimensions (i.e., features)
        nll = torch.sum(nll, dim=-1)

        # Apply the reduction (mean, sum, or no reduction)
        if self.reduction == 'mean':
            return torch.mean(nll)
        elif self.reduction == 'sum':
            return torch.sum(nll)
        else:
            return nll


    main()

