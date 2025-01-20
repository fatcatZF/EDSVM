import numpy as np 

from sklearn.metrics import roc_auc_score

from stable_baselines3 import PPO, SAC, DQN 

import os 
import glob 
import pickle
import json 
from datetime import datetime 

from environment_util import make_env 



import argparse 

from river import drift 

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

    ## Load Trained models
  
    loaded_models = []

    if (args.policy_type=="dqn"):
        AGENT = DQN 
    elif (args.policy_type=="ppo"):
        AGENT = PPO 
    else:
        AGENT = SAC 

    policy_env_name = args.policy_type + '-' + args.env

    ### Load Trained agent
    agent_path = os.path.join('../agents/', policy_env_name)
    agent = AGENT.load(agent_path) 
    print("Successfully Load Trained Agent.")


    model_folder = os.path.join('../models/EDSVM', policy_env_name)
    print(model_folder)

    model_file_pattern = os.path.join(model_folder, f"pipeline_ssvm_*.pkl")
    matching_models = glob.glob(model_file_pattern)
    if len(matching_models)==0:
        print("No available trained models")
        return 
    for model_path in matching_models:
        with open(model_path, 'rb') as f:
            loaded_models.append(pickle.load(f))
    print(f"Number of trained models: {len(loaded_models)}")



    ## Create Environments
    env0, env1, env2, env3 = make_env(name=args.env)
    print("Successfully create environments")

    result = dict()
    for i in range(len(loaded_models)):
        result[f"edsvm_{i}"] = dict() 

    

    ## Validation Steps


    ## Semantic Drift

    ### AUC

    ### F


    ## Noisy Drift



if __name__ == "__main__":
    main()