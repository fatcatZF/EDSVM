import numpy as np 

from sklearn.metrics import roc_auc_score

from stable_baselines3 import PPO, SAC, DQN 

import os 
import glob 
import pickle
import json 

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

    model_folder = os.path.join('../models/IForest', policy_env_name)
    print(model_folder)

    model_file_pattern = os.path.join(model_folder, f"pipeline_iforest_*.pkl")
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
        result[f"iforest_{i}"] = dict() 

    # Run the evaluations
    auc_semantic_values = []
    auc_noise_values = []

    for i, model in enumerate(loaded_models):
        for j in range(args.n_exp_per_model):

            # Validation
            scores_validation = []
            env_current = env0 
            obs_t, _ = env_current.reset()
            for t in range(1, args.env0_steps+1):
                action_t, _state = agent.predict(obs_t, deterministic=True)
                obs_tplus1, r_tplus1, terminated, truncated, info = env_current.step(action_t)
                transition = np.concatenate([obs_t, obs_tplus1-obs_t]).reshape(1,-1)
                x = np.concatenate([transition, action_t.reshape(1,-1)], axis=1)
                score = -model.decision_function(x)[0] # Anomaly Score
                scores_validation.append(score) 
                done = terminated or truncated
                obs_t = obs_tplus1
                if done:
                    obs_t, _ = env_current.reset()
            scores_validation = np.array(scores_validation)
            mu_valid = np.mean(scores_validation)
            sigma_valid = np.std(scores_validation)

            # Semantic Drift
            scores_semantic = []
            env_current = env1 
            obs_t, _ = env_current.reset()
            total_steps = args.env1_steps + args.env2_steps
            for t in range(1, total_steps+1):
                if t%1000 == 0:
                    print(f"model {i}, experiment {j}, step {t}")
                action_t, _state = agent.predict(obs_t, deterministic=True)
                obs_tplus1, r_tplus1, terminated, truncated, info = env_current.step(action_t)
                transition = np.concatenate([obs_t, obs_tplus1-obs_t]).reshape(1,-1)
                x = np.concatenate([transition, action_t.reshape(1,-1)], axis=1)
                score = -model.decision_function(x)[0] # Anomaly Score
                scores_semantic.append(score) 

                done = terminated or truncated

                obs_t = obs_tplus1
                if done:
                    obs_t, _ = env_current.reset()
                if t==args.env1_steps: ## Environment Drift happens 
                    env_current = env2 
                    obs_t, _ = env_current.reset() 
            
            scores_semantic = np.array(scores_semantic)

            # Noisy Drift (env1, env3)
            scores_noise = []
            env_current = env1 
            obs_t, _ = env_current.reset()
            total_steps = args.env1_steps + args.env3_steps
            for t in range(1, total_steps+1):
                if t%1000 == 0:
                    print(f"model {i}, experiment {j}, step {t}")
                action_t, _state = agent.predict(obs_t, deterministic=True)
                obs_tplus1, r_tplus1, terminated, truncated, info = env_current.step(action_t)
                transition = np.concatenate([obs_t, obs_tplus1-obs_t]).reshape(1,-1)
                x = np.concatenate([transition, action_t.reshape(1,-1)], axis=1)
                score = -model.decision_function(x)[0] # Anomaly Score
                scores_noise.append(score) 

                done = terminated or truncated

                obs_t = obs_tplus1
                if done:
                    obs_t, _ = env_current.reset()
                if t==args.env1_steps: ## Environment Drift happens 
                    env_current = env3 
                    obs_t, _ = env_current.reset() 
            
            scores_noise = np.array(scores_noise)

            scores_validation = (scores_validation-mu_valid)/sigma_valid
            scores_semantic = (scores_semantic-mu_valid)/sigma_valid
            scores_noise = (scores_noise-mu_valid)/sigma_valid

            # Compute AUC
            y_env1 = np.zeros(3000)
            y_env2 = np.ones(3000)
            y_env3 = np.ones(3000)
            y_semantic = np.concatenate([y_env1, y_env2])
            y_noise = np.concatenate([y_env1, y_env3])

            auc_semantic = roc_auc_score(y_semantic, scores_semantic)
            auc_noise = roc_auc_score(y_noise, scores_noise)

            auc_semantic_values.append(auc_semantic)
            auc_noise_values.append(auc_noise)

            result[f"iforest_{i}"][f"exp_{j}"] = {"scores_semantic":scores_semantic.tolist(),
                                                     "auc_semantic":auc_semantic,
                                                     "scores_noise":scores_noise.tolist(),
                                                     "auc_noise":auc_noise}
            
    
    result["auc_semantic_mean"] = np.mean(auc_semantic_values) 
    result["auc_noise_mean"] = np.mean(auc_noise_values)

    result_folder = os.path.join('.',"results", args.env)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder) 

    print("results folder", result_folder)

    result_file = f"iforest-{args.env}.json"

    print("result file: ", result_file)

    result_path = os.path.join(result_folder, result_file)

    print("result path: ", result_path) 

    with open(result_path, 'w') as f:
        json.dump(result, f, separators=(',', ':'))




if __name__=="__main__":
    main()