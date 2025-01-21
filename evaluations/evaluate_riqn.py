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



    # Define Base RIQN Regressor

    class RIQN(nn.Module):
            def __init__(self, feature_len=agent.observation_space.shape[-1],
               gru_size=64, quantile_embedding_dim=128,
               num_quantile_sample = 8, fc1_units = 64):
                   super(RIQN, self).__init__()
                   self.gru_size = gru_size
                   self.quantile_embedding_dim = quantile_embedding_dim
                   self.num_quantile_sample = num_quantile_sample
                   self.feature_len = feature_len
                   self.fc1 = nn.Linear(feature_len, fc1_units)
                   self.gru = nn.GRU(fc1_units, gru_size, batch_first=False) #input shape: [seq_len, n_batch, feature_len]
                   self.dropout = nn.Dropout(p=0.2)
                   self.fc2 = nn.Linear(gru_size, gru_size)
                   self.fc3 = nn.Linear(gru_size, feature_len)
                   self.phi = nn.Linear(self.quantile_embedding_dim, gru_size)

                   # Initialize weights
                   for m in self.modules():
                       if isinstance(m, nn.Linear):
                           nn.init.xavier_normal_(m.weight)

            def forward(self, sequence, h0=None, return_final_hidden=False):
                """
                 args:
                    sequence: [seq_len, batch_size, feature_len]
                    tau: [seq_len, batch_size, quantile_embedding_dim]
                    h0: Initial hidden state: [1, batch_size, gru_size]
                 """
                device = next(self.parameters()).device # get the device of the module
                num_quantiles = self.num_quantile_sample
                quantile_embedding_dim = self.quantile_embedding_dim
                seq_len, batch_size, feature_len = sequence.size()
                tau = torch.rand(seq_len, batch_size, num_quantiles).float().to(device)
                tau_expand = tau.expand(quantile_embedding_dim, seq_len, batch_size, num_quantiles).permute(1,2,3,0)
                #print("tau expand shape: ", tau_expand.size())
                pi_mtx = torch.from_numpy(np.pi * np.arange(0,
                                                quantile_embedding_dim)).expand(seq_len,
                                                                                   batch_size,
                                                                                   num_quantiles,
                                                                                   quantile_embedding_dim).float().to(device)
                #print("pi_mtx shape: ", pi_mtx.size())
                cos_tau = torch.cos(tau_expand * pi_mtx)
                phi = F.relu(self.phi(cos_tau)) # shape: [seq_len, batch_size, num_quantiles, gru_size]
                #print("phi: ", phi.size())
                phi = phi.transpose(2,0) #shape: [num_quantiles, batch_size, seq_len, gru_size]
                #print("phi: ", phi.size())

                x = F.relu(self.fc1(sequence)) # [seq_len, batch_size, fc1_units]

                # Process through GRU
                x, hn = self.gru(x, h0) # new x: [seq_len, batch_size, gru_size]; hn: final hidden state
                x = self.dropout(x)

                x = x + F.relu(self.fc2(x)) # new x: [seq_len, batch_size, gru_size]
                #print("x: ", x.size())
                #print("phi: ", phi.size())
                x = x * phi.transpose(1,2) # new x: [num_quantiles, seq_len, batch_size, gru_size]
                x = x.permute(1,2,0,3) # new x: [seq_len, batch_size, num_quantiles, gru_size]

                x = self.fc3(x) # [seq_len, batch_size, num_quantiles, feature_len]

                # concatenate x and tau (for training)
                tau_unsqueezed = tau.unsqueeze(-1) #shape: [seq_len, batch_size, num_quantiles, 1]
                xAndTau = torch.cat((x, tau_unsqueezed), dim=-1) #shape: [seq_len, batch_size, num_quantiles, feature_len+1]

                if return_final_hidden:
                    return xAndTau, hn.detach()
                else:
                    return xAndTau

    # Load Drift Detector Models
    loaded_models = []
    model_folder = os.path.join("../models/RIQN", policy_env_name)
    model_pattern = os.path.join(model_folder,  f"riqn_*")
    matching_models = glob.glob(model_pattern)  

    print(matching_models) 
    if len(matching_models)==0:
        raise NotImplementedError(f"There is no trained NN ensemble for the environment {args.env}.")
    
    for model_path in matching_models:
        riqn = BaggingRegressor(
             estimator = RIQN,
             n_estimators = 5,
             cuda = False,
            )
        io.load(riqn, model_path)
        loaded_models.append(riqn) 

    
    print(f"Number of trained models: {len(loaded_models)}")



    ## Create environments
    env0, env1, env2, env3 = make_env(name=args.env)
    print("Successfully create environments")

    result = dict()
    for i in range(len(loaded_models)):
        result[f"riqn_{i}"] = dict()

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
                obs_tplus1, r_tplus1, terminated, truncated, info = env0.step(action_t)

                X_val.append(obs_t)
                y_val.append(obs_tplus1)
                done = terminated or truncated

                obs_t = obs_tplus1

                if done: 
                    obs_t, _ = env_current.reset()

            X_val = np.array(X_val)
            y_val = np.array(y_val) 
            X_val_tensor = torch.from_numpy(X_val).unsqueeze(1).float() # Add Batch size = 1
            y_val_tensor = torch.from_numpy(y_val).unsqueeze(1).float() # Add Batch Size = 1
            with torch.no_grad():
                ypAndTau_val = riqn(X_val_tensor)
                yp_val = ypAndTau_val[:,:,:,:-1]

            mu_val = np.mean((yp_val.permute(2,0,1,3) - y_val_tensor).abs().mean(dim=0).squeeze(1).mean(dim=1).numpy())
            std_val = np.std((yp_val.permute(2,0,1,3) - y_val_tensor).abs().mean(dim=0).squeeze(1).mean(dim=1).numpy())

            
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

                X.append(obs_t)
                y.append(obs_tplus1)
                done = terminated or truncated
                obs_t = obs_tplus1

                if done:
                   obs_t, _ = env_current.reset()
                if t==args.env1_steps:
                   env_current = env2
                   obs_t, _ = env_current.reset()

            X = np.array(X)
            y = np.array(y)
            X_tensor = torch.from_numpy(X).unsqueeze(1).float() # Add Batch Size = 1
            y_tensor = torch.from_numpy(y).unsqueeze(1).float() # Add Batch Size = 1

            X_env1_tensor = X_tensor[:args.env1_steps]
            y_env1_tensor = y_tensor[:args.env1_steps]

            X_env2_tensor = X_tensor[args.env1_steps:]
            y_env2_tensor = y_tensor[args.env1_steps:]

            with torch.no_grad():
                ypAndTau_env1 = riqn(X_env1_tensor)
                yp_env1 = ypAndTau_env1[:,:,:,:-1]
                ypAndTau_env2 = riqn(X_env2_tensor)
                yp_env2 = ypAndTau_env2[:,:,:,:-1]

            error_env1 = (yp_env1.permute(2,0,1,3) - y_env1_tensor).abs().mean(dim=0).squeeze(1).mean(dim=1).detach().numpy()
            error_env2 = (yp_env2.permute(2,0,1,3) - y_env2_tensor).abs().mean(dim=0).squeeze(1).mean(dim=1).detach().numpy()

            scores_semantic = (np.concatenate((error_env1, error_env2), axis=0)-mu_val)/std_val
            y_env1 = np.zeros(3000)
            y_env2 = np.ones(3000)
            y = np.concatenate([y_env1, y_env2])
            auc_semantic = roc_auc_score(y, scores_semantic)

            auc_semantic_values.append(auc_semantic)


            ## Noisy Drift
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

                X.append(obs_t)
                y.append(obs_tplus1)
                done = terminated or truncated
                obs_t = obs_tplus1

                if done:
                   obs_t, _ = env_current.reset()
                if t==args.env1_steps:
                   env_current = env3
                   obs_t, _ = env_current.reset()

            X = np.array(X)
            y = np.array(y)
            X_tensor = torch.from_numpy(X).unsqueeze(1).float() # Add Batch Size = 1
            y_tensor = torch.from_numpy(y).unsqueeze(1).float() # Add Batch Size = 1

            X_env1_tensor = X_tensor[:args.env1_steps]
            y_env1_tensor = y_tensor[:args.env1_steps]

            X_env3_tensor = X_tensor[args.env1_steps:]
            y_env3_tensor = y_tensor[args.env1_steps:]

            with torch.no_grad():
                ypAndTau_env1 = riqn(X_env1_tensor)
                yp_env1 = ypAndTau_env1[:,:,:,:-1]
                ypAndTau_env3 = riqn(X_env3_tensor)
                yp_env3 = ypAndTau_env3[:,:,:,:-1]

            error_env1 = (yp_env1.permute(2,0,1,3) - y_env1_tensor).abs().mean(dim=0).squeeze(1).mean(dim=1).detach().numpy()
            error_env3 = (yp_env3.permute(2,0,1,3) - y_env3_tensor).abs().mean(dim=0).squeeze(1).mean(dim=1).detach().numpy()

            scores_noise = (np.concatenate((error_env1, error_env3), axis=0)-mu_val)/std_val
            y_env1 = np.zeros(3000)
            y_env3 = np.ones(3000)
            y = np.concatenate([y_env1, y_env3])
            auc_noise = roc_auc_score(y, scores_noise)

            auc_noise_values.append(auc_noise)

            result[f"riqn_{i}"][f"exp_{j}"] = {"scores_semantic":scores_semantic.tolist(),
                                                     "auc_semantic":auc_semantic,
                                                     "scores_noise":scores_noise.tolist(),
                                                     "auc_noise":auc_noise}
            
    result["auc_semantic_mean"] = np.mean(auc_semantic_values)
    result["auc_noise_mean"] = np.mean(auc_noise_values)

    result_folder = os.path.join('.',"results", args.env)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder) 

    print("results folder", result_folder)


    result_file = f"riqn-{args.env}.json"

    print("result file: ", result_file)

    result_path = os.path.join(result_folder, result_file)

    print("result path: ", result_path) 

    with open(result_path, 'w') as f:
        json.dump(result, f, separators=(',', ':'))




if __name__=="__main__":

    class QuantileHuberLoss(nn.Module):
      def __init__(self, kappa = 1.0):
          super(QuantileHuberLoss, self).__init__()
          self.kappa = kappa

      def forward(self, predictionAndTau,labelAndMask):
        """
         Compute the Quantile Huber Loss

         args:
           labelAndMask: [batch_size, seq_len, feature_len+1]
           predictionAndTau: [batch_size, seq_len, num_quantiles, feature_len+1]
        """
        labelAndMask = labelAndMask.transpose(0,1) # shape: [seq_len, batch_size, feature_len+1]
        predictionAndTau = predictionAndTau.transpose(0,1) # shape: [seq_len, batch_size, num_quantiles, feature_len+1]
        label = labelAndMask[:,:,:-1] #shape: [seq_len, batch_size, feature_len]
        mask = labelAndMask[:,:,-1].detach() #shape: [seq_len, batch_size]
        prediction = predictionAndTau[:,:,:,:-1] #shape: [seq_len, batch_size, num_quantiles, feature_len]
        tau = predictionAndTau[:,:,:,-1].detach() #shape: [seq_len, batch_size, num_quantiles]
        seq_len, batch_size, num_quantiles, feature_len = prediction.size()
        label = label.expand(num_quantiles, seq_len, batch_size, feature_len).permute(1,2,0,3)
        # shape: [seq_len, batch_size, num_quantiles, feature_len]

        mask = mask.expand(num_quantiles, feature_len, seq_len, batch_size).permute(2,3,0,1)
        # shape: [seq_len, batch_size, num_quantiles, feature_len]

        # compute temporal difference error
        td_error = label - prediction # [seq_len, batch_size, num_quantiles, feature_len]
        # Compute Huber loss
        huber_loss = torch.where(
            torch.abs(td_error) <= self.kappa,
            0.5 * td_error**2,
            self.kappa * (torch.abs(td_error) - 0.5 * self.kappa)
        )  # [seq_len, batch_size, num_quantiles, feature_len]

        # Compute Quantile Regression Loss
        quantile_loss = (torch.abs(tau.unsqueeze(-1) - (td_error < 0).float()) * huber_loss) * mask.float()
        # shape: [seq_len, batch_size, num_quantiles, feature_len]

        # Mean reduction over quantiles and features
        loss = quantile_loss.mean(dim=2) #Compute the mean over the number of quantiles
        # shape: [seq_len, batch_size, feature_len]


        loss = loss.sum() / (seq_len * batch_size) # Reduce by dividing seq_len * batch_size

        return loss

    main()
