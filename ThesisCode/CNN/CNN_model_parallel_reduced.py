import numpy as np
import torch
import torch.nn as nn
import torchvision
import pyboy as pb
import os
import multiprocessing
from gym_env_v2 import RedGymEnv
import json
import socket


configpath = os.path.join(os.path.dirname(__file__), "config_reduced.json")
with open(configpath, 'r') as file:
    config = json.load(file)
    
loading = config.get("loading")

environment = config.get("environment")

environment["init_state"] = os.path.expanduser(environment["init_state"])
environment["gb_path"] = os.path.expanduser(environment["gb_path"])
params = config.get("params")
hostname = socket.gethostname()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size= 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size= 3, stride=1, padding =1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(params["flat_size"] + params["no_img_size"], params["interdim1"])
        self.fc2 = nn.Linear(params["interdim1"], params["interdim2"])
        self.fc3 = nn.Linear(params["interdim2"], params["outdim"])
        
        
    def forward(self,images, non_images):
        x = torch.tanh(self.conv1(images))
        x = self.pool(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, non_images.to(images.device)], dim= 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent():

    datadim: int = None
    interdim1: int = None
    interdim2: int = None
    outdim: int = None
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
        self.fitness_dict = {}

        #might no longer be relevant
        self.action_sequence = []

        if generation is not None:
            self.generation = generation
        else:
            self.generation = 0 + starting_point
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
            "fitness": self.fitness,
            "fitness_dict": self.fitness_dict
        }
    
    def evaluation_reduction(self):
        state = self.phenotype.state_dict()
        return state

    def from_state(self,dict):
        self.datadim = dict["data"]
        self.interdim1 = dict["one"]
        self.interdim2 = dict["two"]
        self.outdim = dict["out"]
        self.seed_sequence = dict["seed_sequence"]
        self.generation = dict["generation"]
        self.fitness = dict["fitness"]        

    def init_genotype(self):
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.seed_sequence[0])
        return [
        torch.randn(4,3,3,3), # weight kernel conv1
        torch.randn(4), # bias kernel conv1
        torch.randn(8, 4, 3, 3), # weight kernel conv2
        torch.randn(8), #bias kernel conv2 
        torch.randn(self.interdim1,self.datadim), #weights fc1
        torch.randn(self.interdim1), #bias fc1
        torch.randn(self.interdim2, self.interdim1), #weights fc2
        torch.randn(self.interdim2), #bias fc2
        torch.randn(self.outdim, self.interdim2), #weights fc3
        torch.randn(self.outdim) # bias fc3
        ]
        torch.random.set_rng_state(rng_state)

    def build_phenotype(self):
        model = NeuralNetwork()
        model.conv1.weight.data = self.genotype[0]
        model.conv1.bias.data = self.genotype[1]
        model.conv2.weight.data = self.genotype[2]
        model.conv2.bias.data = self.genotype[3]
        model.fc1.weight.data = self.genotype[4]
        model.fc1.bias.data = self.genotype[5]
        model.fc2.weight.data = self.genotype[6]
        model.fc2.bias.data = self.genotype[7]
        model.fc3.weight.data = self.genotype[8]
        model.fc3.bias.data = self.genotype[9]
        return model
    
    def mutate(self,newRn, state_dict, attributes = None):
        model = NeuralNetwork()
        model.load_state_dict(state_dict)
        self.phenotype = model


        rng_state = torch.random.get_rng_state()

        torch.manual_seed(newRn)
        
        if params["use_sigma"]:
            sigma = params["sigma_lb"] + (params["sigma_ub"] - params["sigma_lb"]) * torch.rand(1)
        else:
            sigma = 1

        self.seed_sequence.append(newRn)

        for i in range (self.phenotype.conv1.weight.shape[0]):
            noise = torch.randn_like(self.phenotype.conv1.weight.data[i]) * sigma
            self.phenotype.conv1.weight.data[i] += noise
        
        for i in range (self.phenotype.conv2.weight.shape[0]):
            noise = torch.randn_like(self.phenotype.conv2.weight.data[i]) * sigma
            self.phenotype.conv2.weight.data[i] += noise


        self.phenotype.fc1.weight.data += torch.randn(self.interdim1,self.datadim) * sigma
        self.phenotype.fc1.bias.data += torch.randn(self.interdim1) * sigma
        self.phenotype.fc2.weight.data += torch.randn(self.interdim2, self.interdim1) * sigma
        self.phenotype.fc2.bias.data += torch.randn(self.interdim2) * sigma
        self.phenotype.fc3.weight.data += torch.randn(self.outdim, self.interdim2) * sigma
        self.phenotype.fc3.bias.data += torch.randn(self.outdim) * sigma

        self.generation += 1
        
        torch.random.set_rng_state(rng_state)
        return
    
    def reconstruct(self, state_dict=None, seed_sequence = None, from_sequence = False):
        if not from_sequence:
            model = NeuralNetwork()
            model.load_state_dict(torch.load(replay["model_path"]))
            self.phenotype = model
        else:
            for seed in seed_sequence:
                        rng_state = torch.random.get_rng_state()

                        torch.manual_seed(seed)
                        
                        if params["use_sigma"]:
                            sigma = params["sigma_lb"] + (params["sigma_ub"] - params["sigma_lb"]) * torch.rand(1)
                        else:
                            sigma = 1

                        for i in range (self.phenotype.conv1.weight.shape[0]):
                            noise = torch.randn_like(self.phenotype.conv1.weight.data[i]) * sigma
                            self.phenotype.conv1.weight.data[i] += noise
                        
                        for i in range (self.phenotype.conv2.weight.shape[0]):
                            noise = torch.randn_like(self.phenotype.conv2.weight.data[i]) * sigma
                            self.phenotype.conv2.weight.data[i] += noise


                        self.phenotype.fc1.weight.data += torch.randn(self.interdim1,self.datadim) * sigma
                        self.phenotype.fc1.bias.data += torch.randn(self.interdim1) * sigma
                        self.phenotype.fc2.weight.data += torch.randn(self.interdim2, self.interdim1) * sigma
                        self.phenotype.fc2.bias.data += torch.randn(self.interdim2) * sigma
                        self.phenotype.fc3.weight.data += torch.randn(self.outdim, self.interdim2) * sigma
                        self.phenotype.fc3.bias.data += torch.randn(self.outdim) * sigma


        return


def evaluate(model_state):
    env = RedGymEnv(environment)
    env.reset()
    model = NeuralNetwork().to(device)
    model.load_state_dict(model_state)
    screens, no_img_input = prep_obs(env._get_obs())
    screens = screens.to(device)
    no_img_input = no_img_input.to(device)
    finished = False
    stepSequence = []
    while not finished:
        with torch.no_grad():
            output = model(screens, no_img_input)
            max_value, max_index = torch.max(output, dim=1)
            stepSequence.append(max_index.item())
            input,fitness,finished, fitness_dict = env.step(max_index)
            screens, no_img_input = prep_obs(input) #adjusts the observation to the required data format
            screens = screens.to(device)
            no_img_input = no_img_input.to(device)
    return fitness, fitness_dict


def prep_obs(obs):
    #reshaping and concatentating of different observations
    noimg = torch.cat([
        obs["health"],
        torch.tensor([obs["level"]]), 
        obs["badges"], 
        obs["recent_actions"]], dim=0)
    # "screens" has dimensions [72 H,80 W, 3 C] needs adjusting
    screen = (obs["screens"].permute(2,0,1)).unsqueeze(0)
    screen = screen.float()
    #screen_flat = screen.view(screen.size(0), -1)
    map = obs["map"].unsqueeze(0)
    map_flat = map.view(map.size(0), -1)

    return screen, torch.cat([map_flat, noimg.unsqueeze(0)], dim = 1)

def elitism_selection(population):
    if len(population) > 1:
        population.sort(key = lambda individual: individual.fitness, reverse = True)
        return population[0], population[1]
    else:
        return population[0], population[0]

def save(agent, fitness_values = None, starting_point = None):
    print(agent.seed_sequence)
    #makes directory in case it is not there yet
    os.makedirs("sav", exist_ok=True)
    #saves into directory for less clutter
    
    agent_state = agent.to_state()
    with open(f"sav/{hostname}_CNN_agent_state_gen{agent.generation}_f{agent.fitness}_sigma{params['use_sigma']}.json", "w") as file:
        json.dump(agent_state, file)
    if fitness_values is not None:
        metadata = {}
        metadata["max_steps"] = environment["max_steps"]
        metadata["population_size"] = params["population_size"]
        metadata["total_generations"] = params["generations"] + starting_point
        fitness_values['metadata'] = metadata
        with open(f"sav/{hostname}_CNN_fitness_values_by_generation_total_{metadata['total_generations']}.json", "w") as file:
            json.dump(fitness_values, file)
    return

# Logic for params["selection_logic"]: 0 = select best including parent, 1 = select best 2 including parent, 
# 2 = select best excluding parent, 3 select best 2 excluding parent

def generate_offspring(elite):
    offspring = []
    for j in range(params["population_size"]):
            parent = 0
            if params["selection_logic"] == 1 or params["selection_logic"] == 3:
                parent = torch.randint(0,1,size=(1,)).item()
            model = Agent(
                        params["flat_size"] + params["no_img_size"],
                        params["interdim1"],
                        params["interdim2"],
                        params["outdim"],
                        seed_sequence = elite[parent].seed_sequence,
                        generation = i + starting_point,
                        genotype=elite[parent].genotype
                        )
            model.mutate(torch.randint(-2**31, 2**31 - 1, (1,)).item(), elite[parent].evaluation_reduction())
            offspring.append(model)
    #new_population.append(elite)
    return offspring

def parallel_evaluation(population):
    with multiprocessing.Pool(processes=environment["cpu_cores"]) as pool:
        results = pool.map(evaluate, [model.evaluation_reduction() for model in population])

        for model, fitness in zip(population, results):
            model.fitness = fitness[0]
            model.fitness_dict = fitness[1]


    return
if __name__ == "__main__":
    

    multiprocessing.set_start_method("spawn")
    
    
    population = []
    init_seeds = []
    fitness_vectors = {}
    elite = []
    starting_point = 0
    if loading["from_scratch"]:
        parent_fitness = [0, 0]
        torch.manual_seed(params["seed"])
        for i in range (10):
            init_seeds.append(torch.randint(-69420, 69420, (1,)).item())
        for i in range (10):
            model = Agent(
                            params["flat_size"] + params["no_img_size"],
                            params["interdim1"],
                            params["interdim2"],
                            params["outdim"],
                            seed_sequence= [init_seeds[i]])
            model.genotype = model.init_genotype()
            model.phenotype = model.build_phenotype()
            population.append(model)
    else:
        with open (f'sav/{loading["json"]}', "r") as file:
            model_state = json.load(file)
        starting_point = model_state["generation"]
        torch.manual_seed(params["seed"])
        loaded_model = Agent(
            params["flat_size"] + params["no_img_size"],
            params["interdim1"],
            params["interdim2"],
            params["outdim"],
            model_state["seed_sequence"]
            )
        parent_fitness = [0,0]
        loaded_model.genotype = loaded_model.init_genotype()
        loaded_model.phenotype = loaded_model.build_phenotype()
        loaded_model.reconstruct(seed_sequence=loaded_model.seed_sequence[1:], from_sequence= True)
        population.append(loaded_model)

    #Evaluates the model performances

    for i in range (params["generations"]):
        fitness_list = []
        parallel_evaluation(population)
        for model in population:
            fitness_list.append(model.fitness)
        fitness_vectors[f"Generation{i + starting_point}"] = fitness_list

        # Logic for params["selection_logic"]: 0 = select best including parent, 1 = select best 2 including parent, 
        # 2 = select best excluding parent 5% of the time, 3 select best 2 excluding either parent 5% of the time
        if params["selection_logic"] == 0 or params["selection_logic"] == 2:
            if len(elite) < 1:
                elite1, _ = elitism_selection(population)
                elite.append(elite1)
            else:
                if params["selection_logic"] == 2:
                    rn = torch.randint(1,100,(1,)).item()
                    if rn < 95:
                        population.append(elite[0])
                else:
                    population.append(elite[0])
                elite[0], _ = elitism_selection(population)
        elif params["selection_logic"] == 1 or params["selection_logic"] == 3:
            if len(elite) < 1:
                elite1, elite2 = elitism_selection(population)
                elite.append(elite1)
                elite.append(elite2)
            else:
                if params["selection_logic"] == 3:
                    rn1 = torch.randint(1, 100, size= (1,)).item()
                    rn2 = torch.randint(1,100, (1,)).item()
                    if rn1 < 95:
                        population.append(elite[0])
                    if rn2 < 95:
                        population.append(elite[1])
                else:
                    population.append(elite[0])
                    population.append(elite[1])
                elite[0], elite[1] = elitism_selection(population)
        else:
            print("unexpected value for params['selection_logic']")

        #checks for increases along the way and saves the new elites so they can be reviewed
        if (elite[0].fitness > parent_fitness[0]):
            save(elite[0])
        parent_fitness[0] = elite[0].fitness
        if len(elite) > 1 and elite[1].fitness > parent_fitness[1]:
            save(elite[1])
            parent_fitness[1] = elite[1].fitness        
        #torch.save(elite.phenotype.state_dict(), "model_weights_CNN.pth")
        for individual in elite:
            print(f'Fitness: {individual.fitness}, Generation: {individual.generation}')
        population = generate_offspring(elite)
    save(elite, fitness_vectors,starting_point = starting_point)
    
        