import numpy as np
import torch
import torch.nn as nn
import torchvision
import pyboy as pb
import os

init_dict ={
    "datadim" : 4*160*144,
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
        x = x.view(x.size(0),-1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent():
    def __init__(self, data, one, two, out, rn):
        self.datadim = data
        self.interdim1 = one
        self.interdim2 = two
        self.outdim = out
        self.seed = rn
        self.genotype = init_genotype()
        self.phenotype = build_phenotype()

    #THIS WAS BROKEN FIXED IN V3
    def init_genotype(self):
        torch.manual_seed(self.seed[0])
        return [torch.randn(self.datadim), #weights input layer
        torch.randn(self.interdim1), #bias fc1
        torch.randn(self.interdim1), #weights fc1
        torch.randn(self.interdim2), #bias fc2
        torch.randn(self.interdim2), #weights fc2
        torch.randn(self.outdim), # weights output
        torch.randn(self.outdim) # bias output
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
    
    def mutate(self):
        
        return x

def main():
    #sets the Seed specified up top and creates an instance of the model
    init_rn = torch.manual_seed(init_dict["seed"])
    model = NeuralNetwork()
    rom_path = os.path.expanduser('~/rom/PokemonRed.gb')
    pyboy = pb.PyBoy(rom_path, window = "SDL2")
    with open(os.path.expanduser('~/rom/PokemonRed.gb.state'),"rb") as f:
        pyboy.load_state(f)
    #frame_count = 0
    for i in range (600):
        pyboy.tick()
        #frame_count += 1
        if i % 24 == 0:
            screenshot = pyboy.screen.image
            image_tensor = transform(screenshot)
            image_tensor = image_tensor.unsqueeze(0)
            with torch.no_grad():
        #model magic
                output = model(image_tensor)
                max_value, max_index = torch.max(output, dim=1)
                print(max_value.item(),max_index.item())
                match max_index:
                    case 0:
                        pyboy.button("a",3)
                        print("pressed a")
                    case 1:
                        pyboy.button("b",3)
                        print("pressed b")
                    case 2:
                        pyboy.button("start",3)
                        print("pressed start")
                    case 3:
                        pyboy.button('select',3)
                        print("pressed select")
                    case 4:
                        pyboy.button('down',3)
                        print("pressed down")
                    case 5:
                        pyboy.button("up",3)
                        print("pressed up")
                    case 6: 
                        pyboy.button("right",3)
                        print("pressed right")
                    case 7:
                        pyboy.button("left",3)
                        print("pressed left")
                    case _:
                        print("no action taken")        
    pyboy.stop()
    return
    
if __name__ == "__main__":
    main()

