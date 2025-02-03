import numpy as np
import torch
import torch.nn as nn
import torchvision
import pyboy as pb
import os
import multiprocessing
from gym_env import RedGymEnv
import json
import pickle


configpath = os.path.join(os.path.dirname(__file__), "config.json")
with open(configpath, 'r') as file:
    config = json.load(file)
    
loading = config.get("loading")

environment = config.get("environment")

environment["init_state"] = os.path.expanduser(environment["init_state"])
environment["gb_path"] = os.path.expanduser(environment["gb_path"])
params = config.get("params")

device = torch.device("cuda" if torch.cuda.is_available() else cpu)
# print(f"Using device: {device}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size= 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size= 3, stride=1, padding =1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(params["flat_size"] + params["recurr"] + params["no_img_size"], params["interdim1"])
        self.fc2 = nn.Linear(params["interdim1"], params["interdim2"])
        self.fc3 = nn.Linear(params["interdim2"], params["outdim"] + params["recurr"])
        
        self.recurrence = torch.zeros(1,params["recurr"])
        
    def forward(self,images, non_images):
        x = torch.tanh(self.conv1(images))
        x = self.pool(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, non_images.to(images.device), self.recurrence.to(images.device)], dim= 1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        self.recurrence = x[:, -80:]
        return x[:,:7]


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
            "fitness": self.fitness
        }
    
    def evaluation_redux(self):
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
        torch.randn(8,3,3,3), # weight kernel conv1
        torch.randn(8), # bias kernel conv1
        torch.randn(16, 8, 3, 3), # weight kernel conv2
        torch.randn(16), #bias kernel conv2 
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
        model.load_state_dict(torch.load(state_dict))
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
            input,fitness,finished,_ = env.step(max_index)
            screens, no_img_input = prep_obs(input) #adjusts the observation to the required data format
            screens = screens.to(device)
            no_img_input = no_img_input.to(device)
    return fitness


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

def eltism_selection(population):
    highest_index, highest_individual = max(
        enumerate(population), key=lambda x: x[1].fitness
    )
    return highest_individual

def save(agent, fitness_values = None, starting_point = None):
    print(agent.seed_sequence)
    #makes directory in case it is not there yet
    os.makedirs("sav", exist_ok=True)
    #saves into directory for less clutter
    
    agent_state = agent.to_state()
    with open(f"sav/CNN_agent_state_gen{agent.generation}_f{agent.fitness}_sigma{params['use_sigma']}.json", "w") as file:
        json.dump(agent_state, file)
    if fitness_values is not None:
        metadata = {}
        metadata["max_steps"] = environment["max_steps"]
        metadata["population_size"] = params["population_size"]
        metadata["total_generations"] = params["generations"] + starting_point
        fitness_values['metadata'] = metadata
        with open(f"sav/CNN_fitness_values_by_generation_total_{metadata['total_generations']}.json", "w") as file:
            json.dump(fitness_values, file)
    return

def generate_new_population(elite):
    new_population = []
    for j in range(params["population_size"] -1):
            model = Agent(
                        params["flat_size"] + params["recurr"] + params["no_img_size"],
                        params["interdim1"],
                        params["interdim2"],
                        params["outdim"] + params["recurr"],
                        seed_sequence = elite.seed_sequence,
                        generation = i + starting_point,
                        genotype=elite.genotype
                        )
            model.mutate(torch.randint(-2**31, 2**31 - 1, (1,)).item(), "model_weights_CNN.pth")
            new_population.append(model)
    new_population.append(elite)
    return new_population

def parallel_evaluation(population):
    with multiprocessing.Pool(processes=environment["cpu_cores"]) as pool:
        results = pool.map(evaluate, [model.evaluation_redux() for model in population])

        for model, fitness in zip(population, results):
            model.fitness = fitness 


    # for model in population:
    #         env.reset()
    #         model.fitness, model.action_sequence, newSeed = evaluate(env, model)
    #         fitness_list.append(model.fitness)
    return
if __name__ == "__main__":
    
    #num_evns = environment["cpu_cores"]
    #environments = [RedGymEnv(environment, i)for i in range (num_evns)]
    #env = RedGymEnv(environment)
    multiprocessing.set_start_method("spawn")
    
    
    population = []
    init_seeds = []
    fitness_vectors = {}
    starting_point = 0
    if loading["from_scratch"]:
        parent_fitness = 0
        torch.manual_seed(params["seed"])
        for i in range (10):
            init_seeds.append(torch.randint(-69420, 69420, (1,)).item())
        for i in range (10):
            model = Agent(
                            params["flat_size"] + params["recurr"] + params["no_img_size"],
                            params["interdim1"],
                            params["interdim2"],
                            params["outdim"] + params["recurr"],
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
            params["flat_size"] + params["recurr"] + params["no_img_size"],
            params["interdim1"],
            params["interdim2"],
            params["outdim"] + params["recurr"],
            model_state["seed_sequence"]
            )
        parent_fitness = loaded_model.fitness
        loaded_model.genotype = loaded_model.init_genotype()
        loaded_model.phenotype = loaded_model.build_phenotype()
        loaded_model.reconstruct(seed_sequence=loaded_model.seed_sequence[1:], from_sequence= True)
        population.append(loaded_model)

    #Evaluates the model performances

    for i in range (params["generations"]):
        fitness_list = []
        #TODO: parallelize this step:
        parallel_evaluation(population)
        # for model in population:
        #     env.reset()
        #     model.fitness, model.action_sequence, newSeed = evaluate(env, model)
        #     fitness_list.append(model.fitness)
        for model in population:
            fitness_list.append(model.fitness)
        fitness_vectors[f"Generation{i + starting_point}"] = fitness_list
        elite = eltism_selection(population)
        elite.phenotype.recurrence = torch.zeros(1,80)
        #checks for increases along the way and saves the new elites so they can be reviewed
        if (elite.fitness > parent_fitness):
            save(elite)
        parent_fitness = elite.fitness        
        torch.save(elite.phenotype.state_dict(), "model_weights_CNN.pth")
        population = generate_new_population(elite)
        # for j in range(params["population_size"] -1):
        #     model = Agent(
        #                 params["flat_size"] + params["recurr"] + params["no_img_size"],
        #                 params["interdim1"],
        #                 params["interdim2"],
        #                 params["outdim"] + params["recurr"],
        #                 seed_sequence = elite.seed_sequence,
        #                 generation = i + starting_point,
        #                 genotype=elite.genotype
        #                 )
        #     model.mutate(torch.randint(-2**31, 2**31 - 1, (1,)).item(), "model_weights_CNN.pth")
        #     new_population.append(model)
        # new_population.append(elite)
        # population = new_population
        print(f'Fitness: {elite.fitness}, Generation: {elite.generation}')
    save(elite, fitness_vectors,starting_point = starting_point)
    
        