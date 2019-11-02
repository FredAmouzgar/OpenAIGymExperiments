# OpenAIGymExperiments

The code presented here is my pre-experiments with PPO and REINFORCE algorithm before working on image captioning. Special thanks go to Nikhil Barhate as the PPO implementation is built on top of the code provided by <a href="https://github.com/nikhilbarhate99/PPO-PyTorch">nikhilbarhate99</a>.

This code mainly targeted the <a href="https://gym.openai.com/envs/LunarLander-v2/">LunarLander-V2</a> environment from the OpenAI Gym as shown below.

<img src="https://github.com/FredAmouzgar/OpenAIGymExperiments/raw/master/pics/LunarLander.png" width=400 height=300>

How to run the code:
```console
foo@bar:~$ python3 PPO+REINFORCE.py --reinforce_lambda=0 --ppo_lambda=100 --ppo_algorithm={clipped-only|clipped-value|full}
```
<hr>
Credit: <a href="https://arxiv.org/abs/1707.06347">PPO paper</a>, <a href="https://link.springer.com/article/10.1007/BF00992696">REINFORCE paper</a>, <a href="https://github.com/nikhilbarhate99/PPO-PyTorch">PPO implementation</a>
