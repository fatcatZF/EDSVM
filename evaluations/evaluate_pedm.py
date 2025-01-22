import os 
import glob 
import json 

import numpy as np 


from sklearn.metrics import roc_auc_score

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions.multivariate_normal import MultivariateNormal

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
    parser.add_argument("--hidden-dim", type=int, default=500, 
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
      def __init__(self, input_dim=obs_dim+action_dim, hidden_dim=args.hidden_dim, 
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

    ## Create environments
    env0, env1, env2, env3 = make_env(name=args.env)
    print("Successfully create environments")

    result = dict()
    for i in range(len(loaded_models)):
        result[f"pedm_{i}"] = dict()

    ## Run the evaluations
    auc_semantic_values = []
    auc_noise_values = []

    for i, model in enumerate(loaded_models):
        for j in range(args.n_exp_per_model):
           
           # Validation
            X_val = []
            y_val = []
            env_current = env0 
            obs_t, _ = env_current.reset()
            for t in range(1, args.env0_steps+1):
                action_t, _state = agent.predict(obs_t, deterministic=True)
                obs_tplus1, r_tplus1, terminated, truncated, info = env_current.step(action_t)

                X_val.append(np.concatenate((obs_t, action_t.reshape(-1))))
                y_val.append(obs_tplus1-obs_t)
                done = terminated or truncated

                obs_t = obs_tplus1

                if done: 
                    obs_t, _ = env_current.reset()

            X_val = np.array(X_val)
            y_val = np.array(y_val) 
            X_val_tensor = torch.from_numpy(X_val).float() #shape: [batch_size, x_dim]
            y_val_tensor = torch.from_numpy(y_val).float() 

            with torch.no_grad():
               output = pedm(X_val_tensor)
               dim = output.shape[-1] // 2
               mu, var = output[:, :dim], output[:, dim:] #shape: [batch_size, n_features]
               cov = torch.diag_embed(var) # shape: [batch_size, n_features, n_features]
               distrib = MultivariateNormal(loc=mu, covariance_matrix=cov)
               sampled_predictions = distrib.rsample(sample_shape=torch.Size([200]))
               # shape: [200, batch_size, n_features]
               mses = ((sampled_predictions-y_val_tensor)**2).mean(dim=-1).mean(dim=0) #shape[batch_size]
               mu_val = mses.mean().detach().numpy()
               std_val = mses.std().detach().numpy()
            
            
            # Semantic Drift
            X = []
            y = []
            env_current = env1 
            obs_t, _ = env_current.reset()
            total_steps = args.env1_steps + args.env2_steps
            for t in range(1, total_steps+1):
                if t%1000 == 0:
                    print(f"step: {t}")
                action_t, _state = agent.predict(obs_t, deterministic=True)
                obs_tplus1, r_tplus1, terminated, truncated, info = env_current.step(action_t)

                X.append(np.concatenate((obs_t, action_t.reshape(-1))))
                y.append(obs_tplus1-obs_t)
                done = terminated or truncated

                obs_t = obs_tplus1

                if done: 
                    obs_t, _ = env_current.reset()
                if t==args.env1_steps:
                   env_current = env2
                   obs_t, _ = env_current.reset()

            X = np.array(X)
            y = np.array(y)
            X_tensor = torch.from_numpy(X).float()
            y_tensor = torch.from_numpy(y).float()
            with torch.no_grad():
               output = pedm(X_tensor)
               dim = output.shape[-1] // 2
               mu, var = output[:, :dim], output[:, dim:] #shape: [batch_size, n_features]
               cov = torch.diag_embed(var) # shape: [batch_size, n_features, n_features]
               distrib = MultivariateNormal(loc=mu, covariance_matrix=cov)
               sampled_predictions = distrib.rsample(sample_shape=torch.Size([200]))
               # shape: [200, batch_size, n_features]
               mses = ((sampled_predictions-y_tensor)**2).mean(dim=-1).mean(dim=0) #shape[batch_size]
            
            scores_semantic = (mses.detach().numpy()-mu_val)/std_val   
            y_env1 = np.zeros(args.env1_steps)
            y_env2 = np.ones(args.env2_steps)
            y = np.concatenate([y_env1, y_env2])
            auc_semantic = roc_auc_score(y, scores_semantic)
            auc_semantic_values.append(auc_semantic)


            # Noise Drift
            X = []
            y = []
            env_current = env1 
            obs_t, _ = env_current.reset()
            total_steps = args.env1_steps + args.env3_steps
            for t in range(1, total_steps+1):
                if t%1000 == 0:
                    print(f"step: {t}")
                action_t, _state = agent.predict(obs_t, deterministic=True)
                obs_tplus1, r_tplus1, terminated, truncated, info = env_current.step(action_t)

                X.append(np.concatenate((obs_t, action_t.reshape(-1))))
                y.append(obs_tplus1-obs_t)
                done = terminated or truncated

                obs_t = obs_tplus1

                if done: 
                    obs_t, _ = env_current.reset()
                if t==args.env1_steps:
                   env_current = env3
                   obs_t, _ = env_current.reset()

            X = np.array(X)
            y = np.array(y)
            X_tensor = torch.from_numpy(X).float()
            y_tensor = torch.from_numpy(y).float()
            with torch.no_grad():
               output = pedm(X_tensor)
               dim = output.shape[-1] // 2
               mu, var = output[:, :dim], output[:, dim:] #shape: [batch_size, n_features]
               cov = torch.diag_embed(var) # shape: [batch_size, n_features, n_features]
               distrib = MultivariateNormal(loc=mu, covariance_matrix=cov)
               sampled_predictions = distrib.rsample(sample_shape=torch.Size([200]))
               # shape: [200, batch_size, n_features]
               mses = ((sampled_predictions-y_tensor)**2).mean(dim=-1).mean(dim=0) #shape[batch_size]
            
            scores_noise = (mses.detach().numpy()-mu_val)/std_val   
            y_env1 = np.zeros(args.env1_steps)
            y_env3 = np.ones(args.env2_steps)
            y = np.concatenate([y_env1, y_env3])
            auc_noise = roc_auc_score(y, scores_noise)
            auc_noise_values.append(auc_noise)

            result[f"pedm_{i}"][f"exp_{j}"] = {"scores_semantic":scores_semantic.tolist(),
                                                     "auc_semantic":auc_semantic,
                                                     "scores_noise":scores_noise.tolist(),
                                                     "auc_noise":auc_noise}
    
    result["auc_semantic_mean"] = np.mean(auc_semantic_values)
    result["auc_noise_mean"] = np.mean(auc_noise_values)

    result_folder = os.path.join('.',"results", args.env)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder) 
    
    print("results folder", result_folder)
    result_file = f"pedm-{args.env}.json"

    print("result file: ", result_file)

    result_path = os.path.join(result_folder, result_file)

    print("result path: ", result_path) 

    with open(result_path, 'w') as f:
        json.dump(result, f, separators=(',', ':'))
           


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

