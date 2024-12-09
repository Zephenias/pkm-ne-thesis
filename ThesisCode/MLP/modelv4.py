import numpy as np
import torch
import torch.nn as nn
import torchvision
import pyboy as pb
import os
from gym_env import RedGymEnv

init_dict ={
    "datadim" : 19597, #obs.shape[1]
    "interdim1" : 2500,
    "interdim2" : 150,
    "outdim" : 7,
    "seed" : 42}

transform = torchvision.transforms.ToTensor()

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(init_dict["datadim"], init_dict["interdim1"])
        self.fc2 = nn.Linear(init_dict["interdim1"], init_dict["interdim2"])
        self.fc3 = nn.Linear(init_dict["interdim2"], init_dict["outdim"])
        
    def forward(self,x):
        # x = x.view(x.size(0),-1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent():
    def __init__(self, data, one, two, out, basern):
        self.datadim = data
        self.interdim1 = one
        self.interdim2 = two
        self.outdim = out
        self.seed = []
        self.seed.append(basern)

    def init_genotype(self):
        torch.manual_seed(self.seed[0])
        return [torch.randn(self.interdim1,self.datadim), #weights fc1
        torch.randn(self.interdim1), #bias fc1
        torch.randn(self.interdim2, self.interdim1), #weights fc2
        torch.randn(self.interdim2), #bias fc2
        torch.randn(self.outdim, self.interdim2), #weights fc3
        torch.randn(self.outdim) # bias fc3
        ]

    def build_phenotype(self):
        model = NeuralNetwork()
        model.fc1.weight.data = self.genotype[0]
        model.fc1.bias.data = self.genotype[1]
        model.fc2.weight.data = self.genotype[2]
        model.fc2.bias.data = self.genotype[3]
        model.fc3.weight.data = self.genotype[4]
        model.fc3.bias.data = self.genotype[5]
        return model
    
    def mutate(self,newRn):
        torch.manual_seed(newRn)
        mutations = [torch.randn(self.interdim1,self.datadim), #weights fc1
        torch.randn(self.interdim1), #bias fc1
        torch.randn(self.interdim2, self.interdim1), #weights fc2
        torch.randn(self.interdim2), #bias fc2
        torch.randn(self.outdim, self.interdim2), #weights fc3
        torch.randn(self.outdim) # bias fc3
        ]
        for i in range(len(self.genotype)):
            self.genotype[i] += mutations[i]
        self.phenotype = self.build_phenotype()
        return



def evaluate(env, model):
    input = prep_obs(env._get_obs())
    finished = False
    while not finished:
        with torch.no_grad():
            output = model.phenotype(input)
            max_value, max_index = torch.max(output, dim=1)
            step_out = env.step(max_index)
            input = prep_obs(step_out[0])
            finished = step_out[2]
            print(env.step_count)
    return



def main():
    #sets the Seed specified up top and creates an instance of the model
    init_rn = torch.manual_seed(init_dict["seed"])
    model = Agent(init_dict["datadim"], init_dict["interdim1"], init_dict["interdim2"], init_dict["outdim"], init_dict["seed"])
    model.genotype = model.init_genotype()
    model.phenotype = model.build_phenotype()
    for i in range (runs):
        evaluateRun(model)
        model.mutate(torch.randint(-2**31, 2**31 - 1, (1,)).item())
    return

def prep_obs(obs):
    #reshaping and concatentating of different observations
    noimg = torch.cat([
        obs["health"],
        torch.tensor([obs["level"]]), 
        obs["badges"], 
        obs["recent_actions"]], dim=0)

    screen = obs["screens"].unsqueeze(0)
    screen_flat = screen.view(screen.size(0), -1)
    map = obs["map"].unsqueeze(0)
    map_flat = map.view(map.size(0), -1)

    return torch.cat([screen_flat, map_flat, noimg.unsqueeze(0)], dim = 1)


if __name__ == "__main__":
    
    env_config = {
        "headless": False,
        "print_fitness": True,
        "init_state": os.path.expanduser('~/rom/PokemonRed.gb.state'),
        "max_steps" : 50,
        "action_freq": 24,
        "gb_path": os.path.expanduser('~/rom/PokemonRed.gb')
    }

    env = RedGymEnv(env_config)

    model = Agent(init_dict["datadim"], init_dict["interdim1"], init_dict["interdim2"], init_dict["outdim"], init_dict["seed"])
    model.genotype = model.init_genotype()
    model.phenotype = model.build_phenotype()
    for i in range (5):
        env.reset()
        evaluate(env, model)
        model.mutate(torch.randint(-2**31, 2**31 - 1, (1,)).item())