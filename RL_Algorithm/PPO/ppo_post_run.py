import sys 
import os
# Add all modules.
sys.path.append(f'{os.getcwd()}')
sys.path.append(f'{os.getcwd()}/NN_Model')
sys.path.append(f'{os.getcwd()}/RL_Environment')

import json 
import logging 
from rl_nlp_world import *
import utils as U
from utils import suffix as models_to_test
import model_cnn as M
import model_nlp_cnn as MNLP
import model_attention as M_Attn
import torch 
import matplotlib.pyplot as plt 
import copy
import model_nlp as M_Simp
from model_context_manager import Model_Context_Manager as MCM

LOG = None 
def create_result_file(y: int, z: str):
    global LOG 
    logging.basicConfig(filename=f'Results/final_score_d3_{y}{z}', filemode='w', level = logging.INFO, format='%(asctime)3s - %(filename)s:%(lineno)d - %(message)s')
    LOG = logging.getLogger(__name__)

def run_agent(number, opt):
    def policy(state):
        dist, value = model([state])
        action = dist.sample()
        return action.cpu().numpy().item()
    dbg=False
    episodes=1
    env = RlNlpWorld(render_mode="rgb_array", instr_type = instr_type)
    actionArr, rewardArr = [],[]
    for _ in range(episodes):
        cumulative_reward,steps = 0, 0
        state = env.reset(set_no=number, seed=42)
        state = model.pre_process(state)
        while True:
            if dbg==True:
                print(state['text'])
                plt.imshow(state['visual'])
                plt.show()
            action = policy(state)  # User-defined policy function
            next_state, reward, terminated, info = env.step(action)
            next_state = model.pre_process(next_state) 
            actionArr.append(action)
            rewardArr.append(reward)
            state = copy.deepcopy(next_state)
            cumulative_reward += reward
            steps += 1
            if terminated: break
    env.close()
    return {"cumulative_reward":cumulative_reward,"action": actionArr, "reward": rewardArr}

if __name__=='__main__':
    for model_to_test in models_to_test:
        for id, val in enumerate(model_to_test):
            model_to_test[id] = 'model_' + val 
    with open('Configs/test_config.json', 'r') as file:
        args = json.load(file)

    instr_type = "policy" if args["instr_type"] == 0 else "state"
    create_result_file(args["model"], "" if args["instr_type"] == 0 else "s")
    if instr_type == "state":
        for model_to_test in models_to_test:
            for id, model in enumerate(model_to_test):
                model_to_test[id] = model + '_stateInstr'

    with open('Configs/test_path.json','r') as file:
        paths = json.load(file)
    for key, value in paths.items(): 
        if key != models_to_test[args["model"]][args["order"]]: continue
        LOG.info(f'TEST NAME {key}')
        # plot_ppo(f"train_model_d3_{args["model"]}{"" if args["instr_type"] == 0 else "s"}.json",value["output_path"])
        model = MCM(value["model"]["type"]).model
        model.load_state_dict(torch.load(f'{value["model"]["path"]}', \
                                         weights_only = False))
        
        # TRAIN
        train_set, train_dict, avg_cum = [1], {}, 0
        if len(value["train_set"]) != 0:
            with open(f'{value["train_set"]}','r') as file_tr:
                train_set = json.load(file_tr)
        for no in train_set:
            train_dict[no] = run_agent(no,value["model"]["type"]) # Call function {action, reward,cumulative reward}
            avg_cum += train_dict[no]["cumulative_reward"]
            LOG.info(f'[TRAIN] No[{no}] ~ Reward[{train_dict[no]["cumulative_reward"]}]')
        LOG.info(f'[IMP] train {avg_cum/len(train_set)}')
        with open(f'{value["output_path"]}/train_dict.json', 'w') as file:
            json.dump(train_dict,file)
        LOG.info(f'Train Dict {value["output_path"]}/train_dict.json')

        # FULL Test 
        if args["full_test"] == 1:
            full_test = {}
            avg_cum = 0
            for no in range(1, 1000):
                full_test[no] = run_agent(no, value["model"]["type"])
                avg_cum += full_test[no]["cumulative_reward"]
                LOG.info(f'[FULLTEST] No[{no}] ~ Reward[{full_test[no]["cumulative_reward"]}]')
            LOG.info(f'[IMP] test {avg_cum/1000}')
            with open(f'{value["output_path"]}/full_test_dict.json','w') as file:
                json.dump(full_test,file)
            LOG.info(f'Full Test Dict {value["output_path"]}/full_test_dict.json')  
