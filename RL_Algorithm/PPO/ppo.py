import sys 
import os
# Add all modules.
sys.path.append(f'{os.getcwd()}')
sys.path.append(f'{os.getcwd()}/NN_Model')
sys.path.append(f'{os.getcwd()}/RL_Environment')
import numpy as np 
np.random.seed(seed = 2023)
import time
from rl_nlp_world import *
from utils import gen_data, suffix
from model_context_manager import Model_Context_Manager as MCM
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import copy
import time
import math
import logging 
from collections import defaultdict

def RESETS(envs, noX = None, override = False):
    global no_count, rank_train
    if override: return envs.reset(set_no = noX), noX
    # UCB
    c = 2.5
    no_count += 1
    for idx, sample in enumerate(rank_train):
        rank, reward, exploration, no = sample
        rank_train[idx][0] = reward - c * math.sqrt(math.log(no_count)/exploration)
    rank_train.sort()
    rank_train[0][2] += 1
    return envs.reset(set_no = rank_train[0][-1]), no

def STEPS(envs,action):
    return envs.step(action)

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    def swap(tensor, indx_a, indx_b):
        temp = tensor[indx_a].clone()
        tensor[indx_a] = tensor[indx_b].clone()
        tensor[indx_b] = temp.clone()
    tr_size = len(states)
    indx = [ i for i in range(tr_size) ]
    np.random.shuffle(indx)
    for indx_a, indx_b in enumerate(indx):
        swap(actions, indx_a, indx_b)
        swap(log_probs, indx_a, indx_b)
        swap(returns, indx_a, indx_b)
        swap(advantage, indx_a, indx_b)
    lb, ub = -mini_batch_size, 0
    while ub < tr_size:
        lb += mini_batch_size
        ub = min(tr_size, lb + mini_batch_size)
        yield states[lb : ub], actions[lb : ub], log_probs[lb : ub], \
              returns[lb : ub], advantage[lb : ub]
        


def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    global frame_idx
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            loss = 0.5 * critic_loss + actor_loss - 0.1 * entropy
            LOG.debug(f'Shapes RS[{ratio.shape}], NLPS[{new_log_probs.shape}], OLPS[{old_log_probs.shape}], S1S[{surr1.shape}], S2S[{surr2.shape}], AS[{advantage.shape}]')
            LOG.debug(f'TL[{loss.item()}], CL[{critic_loss.item()}], AL[{actor_loss.item()}], EL[{entropy.item()}]')
            if frame_idx % 1000 == 0:
                LOG.info(f'TL[{loss.item()}], CL[{critic_loss.item()}], AL[{actor_loss.item()}], EL[{entropy.item()}]')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def compute_gae(rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [0]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def test_env_with_trainset(model):
    global max_steps_per_episode, device 
    model = model.eval()
    cum_reward = 0
    for no in train_set:
        state, no = RESETS(env, noX = no, override = True)
        state = model.pre_process(state)
        for _ in range(max_steps_per_episode):
            dist, value = model([state])
            action = dist.sample()
            next_state, reward, done, info = STEPS(env,action.item())
            next_state = model.pre_process(next_state)
            state=copy.deepcopy(next_state)
            cum_reward += reward 
            if done: break
    model = model.train()
    return cum_reward/len(train_set)

                
if __name__=='__main__':
    with open('Configs/train_config.json', 'r') as file:
        args = json.load(file)
    logging.basicConfig(level = args["log"], format='%(asctime)3s - %(filename)s:%(lineno)d - %(message)s')
    LOG = logging.getLogger(__name__)
    LOG.warning(f'Params: {args}')
    train_set, eval_set = gen_data(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scale_tr, rank_train = 10, [[0, -1, 1, no] for no in train_set] # [Total_rank, Reward, No_of_times_encountered, no]
    no_count = 0
    max_episodes = max(scale_tr * len(train_set), args["iter"])
    max_steps_per_episode_list=[40, 50, 64, 5] 
    max_steps_per_episode = max_steps_per_episode_list[args["order"]]
    max_frames = max_episodes * max_steps_per_episode
    frame_idx = 0
    instr_type = "policy" if args["instr_type"] == 0 else "state"
    if instr_type == "state":
        for suf in suffix:
            for id, word in enumerate(suf):
                suf[id] = word + '_stateInstr' 
    env = RlNlpWorld(render_mode="rgb_array", instr_type = instr_type)
    lr               = 1e-5
    mini_batch_size  = 16
    ppo_epochs       = 4
    model = MCM(args["model"]).model
    try:
        with open('Configs/test_path.json','r') as file:
            paths = json.load(file)
        model_name = "model_" + suffix[args["model"]][args["order"]]
        model.load_state_dict(torch.load(paths[model_name]["model"]["path"], \
                                         weights_only = False))
    except:
        LOG.warning(f'There is no model to load.')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    test_rewards = []
    _start_time, prev_time = time.time(), time.time()
    action_dict, reward_dict = defaultdict(int), defaultdict(int)
    episodeNo = 0
    n_agents = 4
    while frame_idx < max_frames:
        if time.time() - prev_time > 300: # Every 5 mins
            prev_time=time.time()
            LOG.warning(f'% Exec Left {100 - (frame_idx*100/max_frames)}; \
                        Time Consumed {time.time() - _start_time} sec')
        log_probsArr = []
        valuesArr    = []
        statesArr    = []
        actionsArr   = []
        rewardsArr   = []
        masksArr     = []
        entropy = 0
        state, nos = RESETS(env)
        state = model.pre_process(state)
        episodeNo += 1
        completed = False
        # PARALLEL AGENTS.
        for agent in range(n_agents):
            state, nos = RESETS(env, nos, True)
            state = model.pre_process(state)
            for _iter in range(max_steps_per_episode):
                dist, value = model([state])
                action = dist.sample()
                next_state, reward, done, info = STEPS(env, action.item())
                next_state = model.pre_process(next_state) 
                action_dict[action.item()] += 1
                reward_dict[reward] += 1
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                valuesArr.append(value)
                log_probsArr.append(torch.FloatTensor([log_prob]).unsqueeze(1).to(device))
                rewardsArr.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
                masksArr.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))
                actionsArr.append(torch.FloatTensor([action]).unsqueeze(1).to(device))
                statesArr.append(state)                
                state = copy.deepcopy(next_state)
                if frame_idx % 5000 == 0:
                    LOG.warning(f'Discovery {action_dict}, {reward_dict}')
                if done: 
                    if reward == 1: completed = True 
                    break
            if frame_idx % 50000 == 0:
                test_reward = np.mean([test_env_with_trainset(model) for _ in range(1)])
                test_rewards.append([frame_idx, test_reward])
                with open(f'Results/train_model_d3_{args["model"]}{"" if args["instr_type"] == 0 else "s"}.json', 'w') as file:
                    json.dump(test_rewards, file)
                LOG.warning(f'Saving Model ...')
                torch.save(model.state_dict(),f'Results/model_{suffix[args["model"]][args["order"]]}.ml')            
            frame_idx += 1

        with torch.no_grad():
            temp_var = torch.tensor(rewardsArr).squeeze()
            rank_train[0][1] = temp_var.mean().item()
        returns = compute_gae(rewardsArr, masksArr, valuesArr)
        returns = torch.cat(returns).detach()
        log_probsArr = torch.cat(log_probsArr).detach()
        valuesArr = torch.cat(valuesArr).detach()
        actionsArr = torch.cat(actionsArr)
        advantage = returns - valuesArr
        ppo_update(model, optimizer, ppo_epochs, mini_batch_size, statesArr, \
                   actionsArr, log_probsArr, returns, advantage)

    torch.save(model.state_dict(), f'Results/model_{suffix[args["model"]][args["order"]]}.ml')