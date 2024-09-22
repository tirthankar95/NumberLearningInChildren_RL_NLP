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

OFFSET = 1000

def RESETS(envs, noX = None, override = False):
    global no_count, rank_train
    if override: return envs.reset(set_no = noX), noX
    # UCB
    c = 0.5
    no_count += 1
    for idx, sample in enumerate(rank_train):
        rank, reward, exploration, no = sample
        rank_train[idx][0] = reward - c * math.sqrt(math.log(no_count)/exploration)
    rank_train.sort()
    rank_train[0][2] += 1
    return envs.reset(set_no = rank_train[0][-1]), rank_train[0][-1]

def STEPS(envs,action):
    return envs.step(action)

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    def swap(tensor, indx_a, indx_b):
        is_tensor = True if type(tensor) == torch.Tensor else False
        temp = tensor[indx_a].clone() if is_tensor else copy.deepcopy(tensor[indx_a])
        tensor[indx_a] = tensor[indx_b].clone() if is_tensor else copy.deepcopy(tensor[indx_b])
        tensor[indx_b] = temp.clone() if is_tensor else copy.deepcopy(temp)
    tr_size = len(states)
    indx = [ i for i in range(tr_size) ]
    np.random.shuffle(indx)
    for indx_a, indx_b in enumerate(indx):
        swap(states, indx_a, indx_b)
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
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            optimizer.zero_grad()
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)
            ratio = (new_log_probs - old_log_probs).exp()
            # assert torch.all(advantage >= 0)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss  = -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            loss = 0.5 * critic_loss + actor_loss - 0.01 * entropy
            if frame_idx % OFFSET == 0:
                LOG.info(f'TL[{round(loss.item(), 5)}], CL[{round(critic_loss.item(), 5)}], AL[{round(actor_loss.item(), 5)}], EL[{round(entropy.item(), 5)}]')
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

best_policy_arr, best_policy_indx = [], -1
def best_policy(no = -1):
    global best_policy_arr, best_policy_indx
    if no != -1:
        h, t, u = (no % 1000) // 100, (no % 100) // 10, (no % 10) // 1
        best_policy_arr = [0, 3] * h + [1, 4] * t + [2, 5] * u
        best_policy_indx = -1

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
    mini_batch_size, ppo_epochs, n_agents = 1, 1, 1
    with open('Configs/train_config.json', 'r') as file:
        args = json.load(file)
    logging.basicConfig(level = args["log"], format='%(asctime)3s - %(filename)s:%(lineno)d - %(message)s')
    LOG = logging.getLogger(__name__)
    LOG.warning(f'Params: {args}')
    train_set, eval_set = gen_data(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scale_tr, rank_train = 1, [[0, -1, 1, no] for no in train_set] # [Total_rank, Reward, No_of_times_encountered, no]
    no_count = 0
    max_episodes = max(scale_tr * len(train_set), args["iter"])
    max_steps_per_episode_list = [40, 50, 64, 5] 
    max_steps_per_episode = max_steps_per_episode_list[args["order"]]
    max_frames = max_episodes * max_steps_per_episode
    frame_idx = 0
    instr_type = "policy" if args["instr_type"] == 0 else "state"
    if instr_type == "state":
        for suf in suffix:
            for id, word in enumerate(suf):
                suf[id] = word + '_stateInstr' 
    env = RlNlpWorld(render_mode="rgb_array", instr_type = instr_type)
    lr = 1e-5
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
    while frame_idx < max_frames:
        if time.time() - prev_time > 300: # Every 5 mins
            prev_time=time.time()
            LOG.warning(f'% Exec Left {100 - (frame_idx*100/max_frames)}; Time Consumed {time.time() - _start_time} sec')
        log_probsArr = []
        valuesArr    = []
        statesArr    = []
        actionsArr   = []
        rewardsArr   = []
        masksArr     = []
        entropy = 0
        episodeNo += 1
        completed = False
        reset_seen, seen = 1, {}
        # PARALLEL AGENTS.
        for agent in range(n_agents):
            state, nos = RESETS(env)
            best_policy(nos)
            # EXPONENTIAL BACKOFF
            if frame_idx % reset_seen:
                seen = {}
                reset_seen = 2 * reset_seen + OFFSET
            state = model.pre_process(state)
            for _iter in range(max_steps_per_episode):
                dist, value = model([state])
                action = dist.sample()
                log_prob = dist.log_prob(action)
                if nos not in seen and args["expert_agent"]: 
                    action = torch.tensor(best_policy_arr[_iter])
                    log_prob = torch.tensor([0.0]) # log(1) = 0.0
                    value = torch.tensor([0.0]) # Convert into reinforce algorithm.
                next_state, reward, done, info = STEPS(env, action.item())
                next_state = model.pre_process(next_state) 
                action_dict[action.item()] += 1
                reward_dict[reward] += 1
                entropy += dist.entropy().mean()
                valuesArr.append(value)
                log_probsArr.append(torch.FloatTensor([log_prob]).unsqueeze(1).to(device))
                rewardsArr.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
                masksArr.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))
                actionsArr.append(torch.FloatTensor([action]).unsqueeze(1).to(device))
                statesArr.append(state)                
                state = copy.deepcopy(next_state)
                if frame_idx % OFFSET == 0:
                    LOG.warning(f'Discovery {action_dict}, {reward_dict}')
                if agent == 0: frame_idx += 1
                if done: 
                    if reward == 1: completed = True 
                    break
            seen[nos] = True
            if frame_idx % OFFSET * 10 == 0:
                test_reward = np.mean([test_env_with_trainset(model) for _ in range(1)])
                test_rewards.append([frame_idx, test_reward])
                with open(f'Results/train_model_d3_{args["model"]}{"" if args["instr_type"] == 0 else "s"}.json', 'w') as file:
                    json.dump(test_rewards, file)
                LOG.warning(f'Saving Model ...')
                torch.save(model.state_dict(),f'Results/model_{suffix[args["model"]][args["order"]]}.ml')            

        with torch.no_grad():
            temp_var = torch.tensor(rewardsArr).squeeze()
            rank_train[0][1] = temp_var.mean().item()
        returns = compute_gae(rewardsArr, masksArr, valuesArr)
        returns = torch.cat(returns).detach()
        log_probsArr = torch.cat(log_probsArr).detach()
        valuesArr = torch.cat(valuesArr).detach()
        actionsArr = torch.cat(actionsArr).detach()
        advantage = returns - valuesArr
        ppo_update(model, optimizer, ppo_epochs, mini_batch_size, statesArr, \
                   actionsArr, log_probsArr, returns, advantage)

    torch.save(model.state_dict(), f'Results/model_{suffix[args["model"]][args["order"]]}.ml')


