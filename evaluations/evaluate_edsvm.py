import numpy as np 

from collections import deque

from scipy.stats import ks_2samp

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

def compute_ph_auc(y, score):
    score_cumsum = np.cumsum(score)
    return roc_auc_score(y, score_cumsum)



def compute_adwin_auc(y, score):
     indicators = []
     sliding_window = []

     for t, x_t in enumerate(score):
        sliding_window.append(x_t)
        
        # Apply ADWIN logic: split the window and calculate max |mu_w1-mu_w2|
        best_split = 0
        max_diff = 0

        for split in range(1, len(sliding_window)):
            W1 = sliding_window[:split]
            W2 = sliding_window[split:]

            mu_W1 = sum(W1) / len(W1)
            mu_W2 = sum(W2) / len(W2)
            diff = abs(mu_W1 - mu_W2)
            if diff > max_diff:
                best_split = split

        indicators.append(max_diff)
     
     # Compute AUC
     auc = roc_auc_score(y, indicators)
     return auc 
    


def compute_kswin_auc(y, score, reference_window):
    p_values = []
    test_size = 100
    test_window = deque([], maxlen=test_size)
    for x_t in data_stream:
        test_window.append(x_t)
        # Compute p-value
        if len(test_window) < test_size:
            p = 1.
        else:
            p = ks_2samp(reference_window, test_window).pvalue 
        
        p_values.append(p)

    p_values = np.array(p_values)
    auc = roc_auc_score(y, p_values)
    return auc 







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

    # Run the evaluations
    auc_semantic_values = []
    auc_noise_values = []

    auc_semantic_values_ph = []
    auc_semantic_values_ad = []
    auc_semantic_values_ks = []

    auc_noise_values_ph = []
    auc_noise_values_ad = []
    auc_noise_values_ks = []
    
    ph_delays_semantic = []
    ph_fas_semantic = []
    ph_delays_noise = []
    ph_fas_noise = []
    ad_delays_semantic = []
    ad_fas_semantic = []
    ad_delays_noise = []
    ad_fas_noise = []
    ks_delays_semantic = []
    ks_fas_semantic = []
    ks_delays_noise = []
    ks_fas_noise = []
    delays_semantic = []
    fas_semantic = []
    delays_noise = []
    fas_noise = []

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


            # Semantic Drift (env1, env2)
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

            auc_semantic_ph = compute_ph_auc(y_semantic, scores_semantic)
            auc_semantic_ad = compute_adwin_auc(y_semantic, scores_semantic)
            auc_semantic_ks = compute_kswin_auc(y_semantic, scores_semantic, 
                                scores_validation)

            auc_semantic_values_ph.append(auc_semantic_ph)
            auc_semantic_values_ad.append(auc_semantic_ad)
            auc_semantic_values_ks.append(auc_semantic_ks)

            auc_noise_ph = compute_ph_auc(y_noise, scores_noise)
            auc_noise_ad = compute_adwin_auc(y_noise, scores_noise)
            auc_noise_ks = compute_kswin_auc(y_noise, scores_noise,
                             scores_validation)
            
            auc_noise_values_ph.append(auc_noise_ph)
            auc_noise_values_ad.append(auc_noise_ad)
            auc_noise_values_ks.append(auc_noise_ks)

            auc_semantic_values.append(auc_semantic)
            auc_noise_values.append(auc_noise)
            
            # Page-Hinkley 
            ## Semantic
            ph = drift.PageHinkley(mode="up", delta=0.005)
            fa = 0
            delay = args.env2_steps+1000
            for t, val in enumerate(scores_semantic):
                ph.update(val)
                if ph.drift_detected and val>0:
                    if t < args.env1_steps:
                        fa += 1
                    if t >= args.env1_steps:
                        delay = t - args.env1_steps
                        break 

            ph_delays_semantic.append(delay)
            ph_fas_semantic.append(fa)
            delays_semantic.append(delay)
            fas_semantic.append(fa)

            ## Noise
            ph = drift.PageHinkley(mode="up", delta=0.005)
            fa = 0
            delay = args.env3_steps+1000
            for t, val in enumerate(scores_noise):
                ph.update(val)
                if ph.drift_detected and val>0:
                    if t < args.env1_steps:
                        fa += 1
                    if t >= args.env1_steps:
                        delay = t - args.env1_steps
                        break 

            ph_delays_noise.append(delay)
            ph_fas_noise.append(fa)
            delays_noise.append(delay)
            fas_noise.append(fa)


            #ADWIN
            ## Semantic
            adwin = drift.ADWIN()
            fa = 0
            delay = args.env2_steps+1000
            for t, val in enumerate(scores_semantic):
                adwin.update(val)
                if adwin.drift_detected and val>0:
                    if t<args.env1_steps:
                       fa+=1
                    if t>=args.env1_steps:
                       delay = t-args.env1_steps
                       break
            
            ad_delays_semantic.append(delay)
            ad_fas_semantic.append(fa)
            delays_semantic.append(delay)
            fas_semantic.append(fa)

            ## Noise
            adwin = drift.ADWIN()
            fa = 0
            delay = args.env3_steps+1000
            for t, val in enumerate(scores_noise):
                adwin.update(val)
                if adwin.drift_detected and val>0:
                    if t<args.env1_steps:
                       fa+=1
                    if t>=args.env1_steps:
                       delay = t-args.env1_steps
                       break
            
            ad_delays_noise.append(delay)
            ad_fas_noise.append(fa)
            delays_noise.append(delay)
            fas_noise.append(fa)


            # KSWIN
            ## Semantic
            kswin = drift.KSWIN(window=scores_validation.tolist())
            fa = 0
            delay = args.env2_steps+1000
            for t, val in enumerate(scores_semantic):
                kswin.update(val)
                if kswin.drift_detected and val>0:
                    if t < args.env1_steps:
                        fa += 1
                    if t >= args.env1_steps:
                        delay = t - args.env1_steps

            ks_delays_semantic.append(delay)
            ks_fas_semantic.append(fa)
            delays_semantic.append(delay)
            fas_semantic.append(fa)

            ## Noise
            kswin = drift.KSWIN(window=scores_validation.tolist())
            fa = 0
            delay = args.env3_steps+1000
            for t, val in enumerate(scores_noise):
                kswin.update(val)
                if kswin.drift_detected and val>0:
                    if t < args.env1_steps:
                        fa += 1
                    if t >= args.env1_steps:
                        delay = t - args.env1_steps

            ks_delays_noise.append(delay)
            ks_fas_noise.append(fa)
            delays_noise.append(delay)
            fas_noise.append(fa)




            result[f"edsvm_{i}"][f"exp_{j}"] = {"scores_validation": scores_validation.tolist(),
                                                "scores_semantic":scores_semantic.tolist(),
                                                "auc_semantic":auc_semantic,
                                                "auc_semantic_ph": auc_semantic_ph,
                                                "auc_semantic_ad": auc_semantic_ad,
                                                "auc_semantic_ks": auc_semantic_ks,
                                                "scores_noise":scores_noise.tolist(),
                                                "auc_noise":auc_noise,
                                                 "auc_noise_ph":auc_noise_ph,
                                                 "auc_noise_ad": auc_noise_ad,
                                                 "auc_noise_ks": auc_noise_ks}
            
    
    result["auc_semantic_mean"] = np.mean(auc_semantic_values)
    result["auc_semantic_mean_ph"] = np.mean(auc_semantic_values_ph)
    result["auc_semantic_mean_ad"] = np.mean(auc_semantic_values_ad)
    result["auc_semantic_mean_ks"] = np.mean(auc_semantic_values_ks)

    result["auc_noise_mean"] = np.mean(auc_noise_values)
    result["auc_noise_mean_ph"] = np.mean(auc_noise_values_ph)
    result["auc_noise_mean_ad"] = np.mean(auc_noise_values_ad)
    result["auc_noise_mean_ks"] = np.mean(auc_noise_values_ks)

    result["ph_delays_semantic_mean"] = np.mean(ph_delays_semantic)
    result["ph_delays_noise_mean"] = np.mean(ph_delays_noise)
    result["ph_fas_semantic_mean"] = np.mean(ph_fas_semantic)
    result["ph_fas_noise_mean"] = np.mean(ph_fas_noise)

    result["ad_delays_semantic_mean"] = np.mean(ad_delays_semantic)
    result["ad_delays_noise_mean"] = np.mean(ad_delays_noise)
    result["ad_fas_semantic_mean"] = np.mean(ad_fas_semantic)
    result["ad_fas_noise_mean"] = np.mean(ad_fas_noise)

    result["ks_delays_semantic_mean"] = np.mean(ks_delays_semantic)
    result["ks_delays_noise_mean"] = np.mean(ks_delays_noise)
    result["ks_fas_semantic_mean"] = np.mean(ks_fas_semantic)
    result["ks_fas_noise_mean"] = np.mean(ks_fas_noise)

    result["delays_semantic_mean"] = np.mean(delays_semantic)
    result["delays_noise_mean"] = np.mean(delays_noise)
    result["fas_semantic_mean"] = np.mean(fas_semantic)
    result["fas_noise_mean"] = np.mean(fas_noise)



    result_folder = os.path.join('.',"results", args.env)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder) 

    print("results folder", result_folder)

    result_file = f"edsvm-{args.env}.json"

    print("result file: ", result_file)

    result_path = os.path.join(result_folder, result_file)

    print("result path: ", result_path) 

    with open(result_path, 'w') as f:
        json.dump(result, f, separators=(',', ':'))

    



if __name__ == "__main__":
    main()