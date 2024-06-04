**The Reinforcement Learning Environment**

`Policy type instruction [DEMO]`
![Policy Type Instruction](src/RL_ENV.gif)
---
**Software Architecture**

`During Training`
![Software Architecture during training](src/SA_train.png)
`During Testing`
![Software Architecture during testing](src/SA_test.png)

---
**How to run the code?**

0. Git clone the project 'https://github.com/tirthankarCU/NumberLearningInChildren_ML.git' and go inside one directory.
1. wget http://nlp.stanford.edu/data/glove.6B.zip
   unzip glove*.zip
2. First pull the environment for running the code from docker-hub using. 
   docker pull tirthankar95/rl-nlp:latest
3. sudo docker run -p 8888:8888 -v $(pwd):/NumberLearningInChildren_ML tirthankar95/rl-nlp &
4. sudo docker ps 
   -> Get the name of the image.
5. Go inside the container ~ sudo docker exec -it angry_lewin bash 
   -> Here "angry_lewin" is the name of the container
6. Run from the main directory ~ python3 -W ignore RL_Algorithm/PPO/ppo.py &> Results/console.log &
7. Run from the main directory ~ python3 -W ignore RL_Algorithm/PPO/ppo_post_run.py &
---