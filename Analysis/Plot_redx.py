import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from random import *
import ast 
import sys
import math
sns.set_style('darkgrid')

'''
GLOBAL variables.
'''
image_resolution = 1024
bar_width = 17.5
N_MODELS = 4
prefix = ""
label_arr = ['Model 1: Visual only', 'Model 2: Pretrained', 'Model 3: Language only', 'Model 4: Attention based']
color_arr = ["blue", "green", "orange", "purple"]
seed_arr = ["", "_seed1", "_seed2"]
smooth_factor = 0.5

'''
    Extract reward from final_score*
'''
def extract_hist(filename, type, test_type='[TEST]'):
    def avg(a, b):
        n = len(a)
        for i in range(n): 
            if int(b[i]) == 0: a[i] = 0
            else: a[i] = a[i]/b[i]
    tr, test = [], []
    with open(filename, 'r') as file:
        for line in file:
            extract, extract_t = None, None
            no, reward = None, None
            for word in line.split():
                if word == '[TRAIN]':
                    extract = 1 
                if word == test_type:
                    extract_t = 1
                if extract == 1 or extract_t == 1:
                    if word[0] == 'N':
                        no = int(word[3:-1])
                    if word[0] == 'R':
                        reward = float(word[7:-1])
            if extract == 1: tr.append([no, reward])
            elif extract_t == 1: test.append([no, reward])
        tr_bin, test_bin = [0 for i in range(10)], \
                           [0 for i in range(10)]
        cnt_tr_bin, cnt_test_bin = [0 for i in range(10)], \
                           [0 for i in range(10)]
        if type == 1: # 1 digit.
            x_bin = [x for x in range(0, 10, 1)]
            for x in tr:
                if x[0] > 9: continue
                indx = int(x[0])
                tr_bin[indx] += x[1]
                cnt_tr_bin[indx] += 1
            for x in test:
                if x[0] > 9: continue 
                indx = int(x[0])
                test_bin[indx] += x[1]
                cnt_test_bin[indx] += 1
            avg(tr_bin, cnt_tr_bin)
            avg(test_bin, cnt_test_bin)
            return x_bin, tr_bin, test_bin 
        if type == 2: # 2 digit.
            x_bin = [x for x in range(0, 100, 10)]
            for x in tr:
                if x[0] > 99: continue
                indx = int(x[0]//10)
                tr_bin[indx] += x[1]
                cnt_tr_bin[indx] += 1
            for x in test:
                if x[0] > 99: continue 
                indx = int(x[0]//10)
                test_bin[indx] += x[1]
                cnt_test_bin[indx] += 1
            avg(tr_bin, cnt_tr_bin)
            avg(test_bin, cnt_test_bin)
            return x_bin, tr_bin, test_bin 
        if type == 3: # 3 digit.
            x_bin = [x for x in range(0, 1000, 100)]
            for x in tr:
                if x[0] > 999: continue
                indx = int(x[0]//100)
                tr_bin[indx] += x[1]
                cnt_tr_bin[indx] += 1
            for x in test:
                if x[0] > 999: continue 
                indx = int(x[0]//100)
                test_bin[indx] += x[1]
                cnt_test_bin[indx] += 1
            avg(tr_bin, cnt_tr_bin)
            avg(test_bin, cnt_test_bin)
            return x_bin, tr_bin, test_bin


'''
    Tr curve plot.
'''
def tr_curve():
    suffix_arr = ["", "s"]
    for suffix in suffix_arr:
        plt.figure(figsize=(10,5))
        ############## MODEL 0, 1, 2, 3 ############## 
        for i in range(N_MODELS): # First three models.
            y, x, n = [], [], 0
            for j in range(len(seed_arr)):
                try:
                    with open(f"{prefix}train_model_d3_{i}{suffix}.json{seed_arr[j]}", "r") as file:
                        content = file.read()
                        my_list = ast.literal_eval(content)
                        n = len(my_list)
                        y.append([ my_list[i][1] for i in range(n) ])
                        x.append([ my_list[i][0] for i in range(n) ])
                except Exception as e:
                    print(f'{prefix}train_model_d3_{i}{suffix}.json{seed_arr[j]} missing.')
            xn, y_mean, y_std = [], [], [] # store mean, std.
            lb, avgMx_mean, avgR_prev = 0, 0, None 
            y_sz = len(y) # no of seeds.
            while lb < n: 
                avgR_arr = []
                for y_indx in range(y_sz):
                    value = y[y_indx][lb]
                    avgR_arr.append(value)
                avgR = sum(avgR_arr)/y_sz
                avg_std = math.sqrt(sum([(x-avgR)**2 for x in avgR_arr])/y_sz)
                avgMx_mean = avgR_prev if avgR_prev and avgR_prev - avgR > smooth_factor else avgR
                avgR_prev = avgMx_mean
                xn.append(x[0][lb])
                y_mean.append(avgMx_mean) 
                y_std.append(avg_std)
                lb += 1
            plt.plot(xn, y_mean, linestyle = '-', label = f'{label_arr[i]}', color = f'{color_arr[i]}')
            plt.fill_between(xn, (np.array(y_mean)-np.array(y_std)), \
                                (np.array(y_mean)+np.array(y_std)),\
                                alpha=.2, color = f'{color_arr[i]}')
        plt.xlabel('Epochs')
        plt.ylabel('Cumulative Reward')
        plt.legend(loc = 'upper left', fontsize = 8)
        plt.savefig(f'{prefix}TrInstr1{suffix}.png', dpi = image_resolution)
        plt.show()


'''
    Plot 3 digit numbers state vs policy based.
'''
def digit3_plot():
    ''' 
    Train.
    '''
    suffix_arr = ["", "s"]
    for suffix in suffix_arr:
        plt.figure(figsize=(10,5))
        ############## MODEL 0, 1, 2, 3 ##############     
        for i in range(N_MODELS): 
            color_arr = ["blue", "green", "orange", "purple"]
            label_arr = ["Model 1: Visual only", "Model 2: Pretrained", "Model 3: Language only", "Model 4: Attention based"]
            try:
                x_bin, ytr, ytest = extract_hist(f"{prefix}final_score_d3_{i}{suffix}", 3, test_type="[FULLTEST]")
                plt.bar(np.array(x_bin) + i * bar_width, ytr, bar_width, color = color_arr[i], label = label_arr[i])
                plt.xticks(np.array(x_bin) + bar_width , ("0-99", "100-199", "200-299", "300-399", "400-499", "500-599",
                                                                "600-699", "700-799", "800-899", "900-999"))        
            except Exception as e:
                print(f'{prefix}final_score_d3_{i}{suffix} missing.')
        plt.legend(loc="upper left", fontsize=8)
        plt.xlabel("Upto 3 digit numbers.")
        plt.ylabel("Average reward.")
        plt.title("Train Dataset.")
        plt.savefig(f'{prefix}3Digit_{suffix}_Train.png', dpi=image_resolution)
        plt.show()
        ''' 
        Test.
        '''
        plt.figure(figsize=(10,5))
        for i in range(N_MODELS):
            try:
                x_bin, ytr, ytest = extract_hist(f"{prefix}final_score_d3_{i}{suffix}", 3, test_type="[FULLTEST]")
                plt.bar(np.array(x_bin) + i * bar_width, ytest, bar_width, color = color_arr[i], label = label_arr[i])
                plt.xticks(np.array(x_bin) + bar_width , ("0-99", "100-199", "200-299", "300-399", "400-499", "500-599",
                                                        "600-699", "700-799", "800-899", "900-999"))
            except Exception as e:
                print(f'{prefix}final_score_d3_{i}{suffix} missing.')
        plt.legend(loc = "upper left", fontsize=8)
        plt.xlabel("Upto 3 digit numbers.")
        plt.ylabel("Average reward.")
        plt.title("Test Dataset.")
        plt.savefig(f'{prefix}3Digit_{suffix}_Test.png', dpi=image_resolution)
        plt.show()


'''
    Compare state v/s policy type instructions.
'''
def state_vs_policy():
    global bar_width
    bar_width = 22.5
    ''' 
    Train.
    '''
    plt.figure(figsize=(10,5))
    try:
        x_bin, ytr, ytest = extract_hist(f"{prefix}final_score_d3_3", 3, test_type="[FULLTEST]")
        plt.bar(np.array(x_bin), ytr, bar_width, color="red", label = "Attention Policy based")
        plt.xticks(np.array(x_bin) + bar_width , ("0-99", "100-199", "200-299", "300-399", "400-499", "500-599",
                                                   "600-699", "700-799", "800-899", "900-999"))
    except Exception as e:
        print(f'{prefix}final_score_d3_3 missing.')
    try:
        x_bin, ytr, ytest = extract_hist(f"{prefix}final_score_d3_3_s", 3, test_type="[FULLTEST]")
        plt.bar(np.array(x_bin) + bar_width, ytr, bar_width, color="blue", label = "Attention State based")
        plt.xticks(np.array(x_bin) + bar_width , ("0-99", "100-199", "200-299", "300-399", "400-499", "500-599",
                                                   "600-699", "700-799", "800-899", "900-999"))
    except Exception as e:
        print(f'{prefix}final_score_d3_3_s missing.')
    plt.legend(fontsize=8)
    plt.xlabel("Upto 3 digit numbers.")
    plt.ylabel("Average reward.")
    plt.title("Train Dataset.")
    plt.savefig(f'{prefix}3D_DiffPolicyTrain.png', dpi = image_resolution)
    plt.show()
    ''' 
    Test.
    '''
    plt.figure(figsize=(10,5))
    try:
        x_bin, ytr, ytest = extract_hist(f"{prefix}final_score_d3_3", 3, test_type="[FULLTEST]")
        plt.bar(np.array(x_bin), ytest, bar_width, color="red", label = "Attention Policy based")
        plt.xticks(np.array(x_bin) + bar_width , ("0-99", "100-199", "200-299", "300-399", "400-499", "500-599",
              
                                                   "600-699", "700-799", "800-899", "900-999"))
    except Exception as e:
        print(f'{prefix}final_score_d3_3 missing.')
    try:
        x_bin, ytr, ytest = extract_hist(f"{prefix}final_score_d3_3_s", 3, test_type="[FULLTEST]")
        plt.bar(np.array(x_bin) + bar_width, ytest, bar_width, color="blue", label = "Attention State based")
        plt.xticks(np.array(x_bin) + bar_width , ("0-99", "100-199", "200-299", "300-399", "400-499", "500-599",
                                                   "600-699", "700-799", "800-899", "900-999"))
    except Exception as e:
        print(f'{prefix}final_score_d3_3_s missing.')
    plt.legend(fontsize=8)
    plt.xlabel("Upto 3 digit numbers.")
    plt.ylabel("Average reward.")
    plt.title("Test Dataset.")
    plt.savefig(f'{prefix}3D_DiffPolicyTest.png', dpi = image_resolution)
    plt.show()
    bar_width = 17.5 

'''
    Driver Function.
'''
if __name__ == '__main__':
    tr_curve()
    digit3_plot()
    state_vs_policy()
