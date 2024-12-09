import numpy as np
import torch
import torch.nn as nn
import torchvision
import pyboy as pb
import os

datadim = 4*160*144
interdim1 = 2500
interdim2 = 150
outdim = 7
seed = 47
transform = torchvision.transforms.ToTensor()
max_index = 7


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(datadim, interdim1)
        self.fc2 = nn.Linear(interdim1, interdim2)
        self.fc3 = nn.Linear(interdim2, outdim)
        
    def forward(self,x):
        x = x.view(x.size(0),-1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    #sets the Seed specified up top and creates an instance of the model
    press_key = False
    torch.manual_seed(seed)
    model = NeuralNetwork()
    rom_path = os.path.expanduser('~/rom/PokemonRed.gb')
    pyboy = pb.PyBoy(rom_path, window = "SDL2")
    with open(os.path.expanduser('~/rom/PokemonRed.gb.state'),"rb") as f:
        pyboy.load_state(f)
    frame_count = 0
    for i in range (600):
        if press_key:
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
                    print("pressed up")
                case 5:
                    pyboy.button("up",3)
                    print("pressed down")
                case 6: 
                    pyboy.button("right",3)
                    print("pressed right")
                case 7:
                    pyboy.button("left",3)
                    print("pressed left")
                case 8:
                    print("no action taken")
        press_key = False
        pyboy.tick()
        frame_count += 1
        if frame_count % 24 == 0:
            screenshot = pyboy.screen.image
            image_tensor = transform(screenshot)
            image_tensor = image_tensor.unsqueeze(0)
            with torch.no_grad():
        #model magic
                output = model(image_tensor)
                max_value, max_index = torch.max(output, dim=1)
                print(max_value.item(),max_index.item())
                press_key = True        
    pyboy.stop()
    return

main()