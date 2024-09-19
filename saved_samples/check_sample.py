import numpy as np 
import json 

def get_sentence(tokens):
    t = tokens.squeeze()
    word_arr = []
    for tt in t:
        word_arr.append(rvocab[int(tt.item())])
    sen = " ".join(word_arr)
    return sen
    
# Only check for simple model.
if __name__ == "__main__":
    sno = 1 
    state = np.load(f"sample{sno}/state.npy", allow_pickle = True)
    action = np.load(f"sample{sno}/action.npy", allow_pickle = True)
    returns = np.load(f"sample{sno}/return.npy", allow_pickle = True)
    advantage = np.load(f"sample{sno}/advantage.npy", allow_pickle = True)

    with open("../NN_Model/model_data/token.json") as file:
        vocab = json.load(file)
    rvocab = [""] * len(vocab)
    for k, v in vocab.items():
        rvocab[v] = k 

    m = action.shape[0]
    for i in range(m):
        print('------------------------------')
        print(f'STATE: {get_sentence(state[i]["text"])}')
        print(f'ACTION: {action[i][0]}')
        print(f'RETURNS: {returns[i][0]}')
        print(f'ADVANTAGE: {advantage[i][0]}')
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n')