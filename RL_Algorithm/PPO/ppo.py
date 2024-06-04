import sys 
import os
# Add all modules.
sys.path.append(f'{os.getcwd()}')
sys.path.append(f'{os.getcwd()}/NN_Model')
sys.path.append(f'{os.getcwd()}/gym-examples')

import numpy as np 
np.random.seed(seed = 2023)
import time
import utils as U
import model_cnn as M
import model_nlp as MNLP
import model_simple as M_Simp
import model_attention as M_Attn
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import copy
import gym 
import gym_examples
import math
import time
import numpy as np 
import logging 


curr_number = -1
def RESETS(envs, noX = None, override=True):
    global train_set_counter, train_set, args, curr_number
    if not override:
        temp = noX if args["ease"]>=0 else 1
        envs.reset(set_no = temp)
    set_number=train_set[train_set_counter] if args["ease"]>=0 else 1
    if episodeNo % time_to_learn == 0: #increment only if time to learn has passed.
        if train_set_counter>=len(train_set)-1:
            train_set_counter=0
        train_set_counter += 1
    curr_number = set_number
    return envs.reset(set_no=set_number)


def STEPS(envs,action):
    return envs.step(action)


def ppo_iter(mini_batch_size, states, statesNlp, actions, log_probs, returns, advantage):
    if args["model"] == 0:
        tr_size = states.size(0)
    elif args["model"] == 1:
        tr_size = states.size(0)
    elif args["model"] == 2:
        tr_size = len(statesNlp)
    elif args["model"] == 3:
        tr_size = len(statesNlp)

    indx = [ i for i in range(tr_size)]
    np.random.shuffle(indx)
    lb = 0
    while lb < tr_size:
        rand_ids = indx[lb: min(lb+mini_batch_size, tr_size)]
        lb += mini_batch_size
        if args["model"] == 0:
            yield states[rand_ids, :], states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        elif args["model"] == 1:
            temp_dict = {}
            for r_indx in rand_ids:
                temp = statesNlpArr[r_indx]
                for key,value in temp.items():
                    if key not in temp_dict:
                        temp_dict[key] = value
                    else:
                        temp_dict[key] = torch.cat((temp_dict[key],value),dim = 0)
            yield states[rand_ids, :], temp_dict, actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        elif args["model"] == 2:
            yield None, statesNlpArr[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        elif args["model"] == 3:
            yield states[rand_ids, :], statesNlpArr[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, statesNlps, actions, log_probs, returns, advantages, clip_param=0.2):
    global frame_idx
    for _ in range(ppo_epochs):
        for state, stateNlp, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, statesNlps, actions, log_probs, returns, advantages):
            
            if args["model"] == 0:
                dist, value = model(state)
            elif args["model"] == 1:
                dist, value = model(state,stateNlp)
            elif args["model"] == 2:
                dist, value = model(stateNlp)
            elif args["model"] == 3:
                dist, value = model(state, stateNlp)

            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)
            LOG.debug(f'[action | dist_logProb | dist_newlogProb] {action[0]} {dist.log_prob(action[0])} {new_log_probs}')
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage
            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            loss = 0.5 * critic_loss + actor_loss - 0.01 * entropy
            LOG.debug(f'Shapes RS[{ratio.shape}], NLPS[{new_log_probs.shape}], OLPS[{old_log_probs.shape}], S1S[{surr1.shape}], S2S[{surr2.shape}], AS[{advantage.shape}]')
            LOG.debug(f'TL[{loss.item()}], CL[{critic_loss.item()}], AL[{actor_loss.item()}], EL[{entropy.item()}]')
            if frame_idx % 1000 == 0:
                LOG.info(f'TL[{loss.item()}], CL[{critic_loss.item()}], AL[{actor_loss.item()}], EL[{entropy.item()}]')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def test_env(model):
    global max_steps_per_episode, device 
    cum_reward=0
    for no in train_set:
        state = RESETS(env, noX = no, override=False)
        state["visual"] = U.pre_process(state)
        if args["model"] == 1:
            state["text"] = U.pre_process_text(model,state)
        elif args["model"] == 2:
            state['text'] = model.pre_process([state['text']])
        elif args["model"] == 3:
            state["text"] = model.pre_process([state["text"]])
        
        for _ in range(max_steps_per_episode):
            if args["model"] == 0:
                dist, value = model(state["visual"])
            elif args["model"] == 1:
                dist, value = model(state['visual'],state['text'])  
            elif args["model"] == 2:
                dist, value = model(state['text']) 
            elif args["model"] == 3:
                dist, value = model(state["visual"], state["text"])

            action = dist.sample()
            next_state, reward, done, info = STEPS(env,action.item())
            next_state["visual"] = U.pre_process(next_state)
            if args["model"] == 1:
                next_state["text"] = U.pre_process_text(model,next_state)
            elif args["model"] == 2:
                next_state['text'] = model.pre_process([next_state['text']]) 
            elif args["model"] == 3:
                next_state["text"] = model.pre_process([next_state["text"]])

            state=copy.deepcopy(next_state)
            cum_reward += reward 
            if done: break
    return cum_reward/len(train_set)

                
if __name__=='__main__':
    suffix = [['easy','medium','hard','naive'], \
              ['fnlp_easy','fnlp_medium','fnlp_hard','fnlp_naive'], \
              ['onlp_easy','onlp_medium','onlp_hard','onlp_naive'], \
              ['anlp_easy','anlp_medium','anlp_hard','anlp_naive']]
    with open('Configs/train_config.json', 'r') as file:
        args = json.load(file)
    logging.basicConfig(level = args["log"], format='%(asctime)3s - %(filename)s:%(lineno)d - %(message)s')
    LOG = logging.getLogger(__name__)
    LOG.warning(f'Params: {args}')
    train_set,test_set=U.gen_data(args)
    train_set_counter=0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    time_to_learn, scale_tr = 10, 50
    max_episodes = max(scale_tr*time_to_learn*len(train_set),args["iter"])
    LOG.info(f'Number of Episodes Tr[{len(train_set)}]*{time_to_learn} = {max_episodes}')
    max_steps_per_episode_list=[25,50,100,5] # my_estimation
    max_steps_per_episode = max_steps_per_episode_list[args["ease"]]
    max_frames = max_episodes * max_steps_per_episode
    frame_idx = 0
    early_stopping = False
    '''
    FOR NEW TYPE OF INSTRUCTION (START)
    '''
    instr_type = "policy" if args["instr_type"] == 0 else "state"
    if instr_type == "state":
        for suf in suffix:
            for id, word in enumerate(suf):
                suf[id] = word + '_stateInstr' 
    '''
    FOR NEW TYPE OF INSTRUCTION (END)
    '''
    env = gym.make('gym_examples/RlNlpWorld-v0',render_mode="rgb_array", instr_type = instr_type)
    # max_advantage = 20
    # Neural Network Hyper params:
    lr               = 1e-5
    mini_batch_size  = 1
    ppo_epochs       = 1
    if args["model"] == 0: # Naive model
        model = M.NNModel().to(device) 
    elif args["model"] == 1: # NLP CNN model
        model = MNLP.NNModelNLP().to(device)
    elif args["model"] == 2:
        model = M_Simp.NN_Simple().to(device)
    elif args["model"] == 3:
        img_shape = (3, 224, 224)
        model = M_Attn.NNAttention(img_shape).to(device)

    threshold_reward = env.threshold_reward
    optimizer = optim.Adam(model.parameters(), lr=lr)
    test_rewards = []
    
    _start_time = time.time()
    prev_time=time.time()
    action_dict, reward_dict = {}, {}
    episodeNo = 0
    dict = {}
    while frame_idx < max_frames:
        if early_stopping: break
        if time.time()-prev_time>300: # Every 5 mins
            prev_time=time.time()
            LOG.warning(f'% Exec Left {100-(frame_idx*100/max_frames)}; Time Consumed {time.time()-_start_time} sec')
        log_probsArr = []
        valuesArr    = []
        statesArr,statesNlpArr    = [],[]
        actionsArr   = []
        rewardsArr   = []
        masksArr     = []
        entropy = 0
        state = RESETS(env)

        if args["model"] == 0:
            state["visual"] = U.pre_process(state)
        if args["model"] == 1:
            state["visual"] = U.pre_process(state)
            state["text"] = U.pre_process_text(model,state)
        if args["model"] == 2:
            state["text"] = model.pre_process([state["text"]])
        if args["model"] == 3:
            state["visual"] = U.pre_process(state)
            state["text"] = model.pre_process([state["text"]])

        episodeNo += 1
        completed = False
        for _iter in range(max_steps_per_episode):
            
            if args["model"] == 0:
                dist, value = model(state["visual"])
            elif args["model"] == 1:
                dist, value = model(state['visual'],state['text'])
            elif args["model"] == 2:
                dist, value = model(state['text'])
            elif args["model"] == 3:
                dist, value = model(state["visual"], state["text"])

            action = dist.sample()
            if action.item() not in action_dict:
                action_dict[action.item()] = 0
            action_dict[action.item()] += 1
            next_state, reward, done, info = STEPS(env,action.item())
            if reward not in reward_dict:
                reward_dict[reward] = 0
            reward_dict[reward] += 1
            LOG.debug(f'Current action[{action.item()}] reward[{reward}]')
            
            if args["model"] == 0:
                next_state["visual"] = U.pre_process(next_state)
            if args["model"] == 1:
                next_state["visual"] = U.pre_process(next_state)
                next_state["text"] = U.pre_process_text(model,next_state)
            elif args["model"] == 2:
                next_state["text"] = model.pre_process([next_state["text"]])
            elif args["model"] == 3:
                next_state["visual"] = U.pre_process(next_state)
                next_state["text"] = model.pre_process([next_state["text"]])
            
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            valuesArr.append(value)
            log_probsArr.append(torch.FloatTensor([log_prob]).unsqueeze(1).to(device))
            rewardsArr.append(torch.FloatTensor([reward]).unsqueeze(1).to(device))
            masksArr.append(torch.FloatTensor([1 - done]).unsqueeze(1).to(device))
            actionsArr.append(torch.FloatTensor([action]).unsqueeze(1).to(device))
            
            if args["model"] == 0:
                statesArr.append(state["visual"])
            elif args["model"] == 1:
                statesArr.append(state["visual"])
                statesNlpArr.append(state["text"])
            elif args["model"] == 2:
                statesNlpArr.append(state["text"])
            elif args["model"] == 3:
                statesArr.append(state["visual"])
                statesNlpArr.append(state["text"])
                
            state = copy.deepcopy(next_state)
            if frame_idx % 5000 == 0:
                LOG.warning(f'Discovery {action_dict}, {reward_dict}')
            if frame_idx % 50000 == 0:
                test_reward = np.mean([test_env(model) for _ in range(1)])
                test_rewards.append([frame_idx,test_reward])
                with open(f'Results/test_reward_list_{suffix[args["model"]][args["ease"]]}.json', 'w') as file:
                    json.dump(test_rewards, file)
                if test_reward > threshold_reward: early_stop = True
            if frame_idx % 50000 == 0:
                LOG.info(f'Saving Model ...')
                torch.save(model.state_dict(),f'Results/model_{suffix[args["model"]][args["ease"]]}.ml')
            frame_idx += 1
            if done: 
                if reward == 1:
                    completed = True 
                break
        if args["model"] == 0:
            _, next_value = model(next_state["visual"])
        elif args["model"] == 1:
            _, next_value = model(next_state["visual"],next_state['text'])
        elif args["model"] == 2:
            _, next_value = model(next_state["text"])
        elif args["model"] == 3:
            _, next_value = model(next_state["visual"],next_state['text'])
        returns = compute_gae(next_value, rewardsArr, masksArr, valuesArr)

        returns   = torch.cat(returns).detach()
        log_probsArr = torch.cat(log_probsArr).detach()
        valuesArr    = torch.cat(valuesArr).detach()
        if args["model"] != 2: # exception
            statesArr = torch.cat(statesArr)
        if args["model"] == 2 or args["model"] == 3: # only nlp  
            statesNlpArr = torch.cat(statesNlpArr)
        actionsArr   = torch.cat(actionsArr)
        advantage = returns - valuesArr
        if instr_type == "state":
            if curr_number in dict and completed and dict[curr_number] > len(actionsArr):
                ppo_epochs += 2
                dict[curr_number] = len(actionsArr)
        ppo_update(model, optimizer, ppo_epochs, mini_batch_size, statesArr, statesNlpArr, actionsArr, log_probsArr, returns, advantage)
    torch.save(model.state_dict(),f'Results/model_{suffix[args["model"]][args["ease"]]}.ml')

