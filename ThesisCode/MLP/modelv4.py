import numpy as np
import torch
import torch.nn as nn
import torchvision
import pyboy as pb
import os
from gym_env import RedGymEnv
import json


configpath = os.path.join(os.path.dirname(__file__), "config.json")
with open(configpath, 'r') as file:
    config = json.load(file)
    
loading = config.get("loading")

environment = config.get("environment")

environment["init_state"] = os.path.expanduser(environment["init_state"])
environment["gb_path"] = os.path.expanduser(environment["gb_path"])
params = config.get("params")



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(params["datadim"], params["interdim1"])
        self.fc2 = nn.Linear(params["interdim1"], params["interdim2"])
        self.fc3 = nn.Linear(params["interdim2"], params["outdim"])
        
    def forward(self,x):
        # x = x.view(x.size(0),-1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent():

    datadim: int = None
    interdim1: int = None
    interdim2: int = None
    outdim: int = None
    #basern: int = None
    seed_sequence: list[int] = None
    generation: int = None

    def __init__(self, data, one, two, out, seed_sequence = None, generation = None, genotype = None, state_dict = None):
        self.datadim = data
        self.interdim1 = one
        self.interdim2 = two
        self.outdim = out
        if seed_sequence:
            self.seed_sequence = seed_sequence.copy()
        if seed_sequence is None:
            print("No seed initialized for Model, setting starting seed to 42.")
            self.seed_sequence.append(42)
        self.fitness = 0

        #might no longer be relevant
        self.action_sequence = []

        if generation is not None:
            self.generation = generation
        else:
            self.generation = 0
        if genotype is not None:
            self.genotype = [gene.clone() for gene in genotype]

    def to_state(self) -> dict:
        return {
            "data": self.datadim,
            "one": self.interdim1,
            "two": self.interdim2,
            "out": self.outdim,
            "seed_sequence": self.seed_sequence,
            "generation": self.generation,
        }

    def from_state(self,dict):
        self.datadim = dict["data"]
        self.interdim1 = dict["one"]
        self.interdim2 = dict["two"]
        self.outdim = dict["out"]
        self.seed_sequence = dict["seed_sequence"]
        self.generation = dict["generation"]        

    def init_genotype(self):
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.seed_sequence[0])
        return [torch.randn(self.interdim1,self.datadim), #weights fc1
        torch.randn(self.interdim1), #bias fc1
        torch.randn(self.interdim2, self.interdim1), #weights fc2
        torch.randn(self.interdim2), #bias fc2
        torch.randn(self.outdim, self.interdim2), #weights fc3
        torch.randn(self.outdim) # bias fc3
        ]
        torch.random.set_rng_state(rng_state)

    def build_phenotype(self):
        model = NeuralNetwork()
        model.fc1.weight.data = self.genotype[0]
        model.fc1.bias.data = self.genotype[1]
        model.fc2.weight.data = self.genotype[2]
        model.fc2.bias.data = self.genotype[3]
        model.fc3.weight.data = self.genotype[4]
        model.fc3.bias.data = self.genotype[5]
        return model
    
    def mutate(self,newRn, state_dict, attributes = None):
        model = NeuralNetwork()
        model.load_state_dict(torch.load(state_dict))
        self.phenotype = model


        rng_state = torch.random.get_rng_state()

        torch.manual_seed(newRn)
        
        self.seed_sequence.append(newRn)

        mutations = [torch.randn(self.interdim1,self.datadim), #weights fc1
        torch.randn(self.interdim1), #bias fc1
        torch.randn(self.interdim2, self.interdim1), #weights fc2
        torch.randn(self.interdim2), #bias fc2
        torch.randn(self.outdim, self.interdim2), #weights fc3
        torch.randn(self.outdim) # bias fc3
        ]
        self.phenotype.fc1.weight.data += mutations[0]
        self.phenotype.fc1.bias.data += mutations[1]
        self.phenotype.fc2.weight.data += mutations[2]
        self.phenotype.fc2.bias.data += mutations[3]
        self.phenotype.fc3.weight.data += mutations[4]
        self.phenotype.fc3.bias.data += mutations[5]

        self.generation += 1
        
        torch.random.set_rng_state(rng_state)
        return



def evaluate(env, model):
    input = prep_obs(env._get_obs())
    finished = False
    stepSequence = []
    while not finished:
        with torch.no_grad():
            output = model.phenotype(input)
            max_value, max_index = torch.max(output, dim=1)
            stepSequence.append(max_index.item())
            input,fitness,finished,_ = env.step(max_index)
            input = prep_obs(input) #adjusts the observation to the required data format
            
    return fitness, stepSequence , model.seed_sequence


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
    print(noimg.shape)
    print(map_flat.shape)

    return torch.cat([screen_flat, map_flat, noimg.unsqueeze(0)], dim = 1)

def eltism_selection(population):
    highest_index, highest_individual = max(
        enumerate(population), key=lambda x: x[1].fitness
    )
    return highest_individual

def save(agent):
    print(agent.seed_sequence)
    #makes directory in case it is not there yet
    os.makedirs("sav", exist_ok=True)
    #saves into directory for less clutter
    torch.save(model.phenotype.state_dict(), f"sav/elite_phenotype_gen{agent.generation}_f{agent.fitness}.pth")
    agent_state = agent.to_state()
    with open(f"sav/agent_state_gen{agent.generation}_f{agent.fitness}.json", "w") as file:
        json.dump(agent_state, file)
    return

if __name__ == "__main__":
    

    env = RedGymEnv(environment)
    population = []
    init_seeds = []
    parent_fitness = 0
    significance = 0.2
    torch.manual_seed(params["seed"])
    for i in range (10):
        init_seeds.append(torch.randint(-69420, 69420, (1,)).item())
    for i in range (10):
        model = Agent(params["datadim"], params["interdim1"], params["interdim2"], params["outdim"], seed_sequence= [init_seeds[i]])
        model.genotype = model.init_genotype()
        model.phenotype = model.build_phenotype()
        population.append(model)
    for i in range (params["generations"]):
        for model in population:
            env.reset()
            model.fitness, model.action_sequence, newSeed = evaluate(env, model)
            # model.seed_sequence.append(newSeed)
        #for model in population:
        #    print(model.fitness, model.action_sequence)
        elite = eltism_selection(population)
        #checks for significant jumps along the way and saves the new elites so they can be reviewed
        if parent_fitness > 0:
            if (elite.fitness - parent_fitness)/parent_fitness >= significance:
                save(elite)
        parent_fitness = elite.fitness        
        torch.save(elite.phenotype.state_dict(), "model_weights_MLP.pth")
        new_population = []
        for j in range(params["population_size"] -1):
            model = Agent(params["datadim"], params["interdim1"], params["interdim2"], params["outdim"], seed_sequence = elite.seed_sequence, generation = i, genotype=elite.genotype)
            model.mutate(torch.randint(-2**31, 2**31 - 1, (1,)).item(), "model_weights_MLP.pth")
            new_population.append(model)
        new_population.append(elite)
        population = new_population
        print(f'Fitness: {elite.fitness}, Generation: {elite.generation}')
    save(elite)
    
    # results = {
    #     "fitness": elite.fitness,
    #     "generation": elite.generation,
    #     "action_sequence": elite.action_sequence
    # }
    # with open("outputtest.json", "w") as file:
    #     json.dump(results, file)
        