import json
from einops import repeat

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent
import pyboy
import torch
import torchvision
from skimage.transform import downscale_local_mean
import numpy as np


from global_map import local_to_global, GLOBAL_MAP_SHAPE

class RedGymEnv (Env):
    def __init__(self,config = None, env_id = None):
        self.headless = config["headless"]
        self.print_fitness = config["print_fitness"]
        self.init_state = config["init_state"]
        self.max_steps = config["max_steps"]
        self.act_freq = config["action_freq"]
        self.frame_stacks = 3
        self.gb_path = ["gb_path"]
        self.reset_count = 0
        self.env_id = env_id
        self.stuck_threshold = config["stuck_threshold"]
        self.is_harsh = config["harsh"]

        self.essential_map_locations= {
            v:i for i,v in enumerate([
                40,0,12,1,13,51,2,54,14,59,60,61,15,3,65
            ])
        }

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_SELECT,
            WindowEvent.PRESS_BUTTON_START
        ]

        self.release_actions =[
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_SELECT,
            WindowEvent.RELEASE_BUTTON_START
        ]

        #with open("events.json") as f:
        #    event_names = json.load(f)
        #self.event_names = event_names

        self.output_shape = (72, 80, self.frame_stacks) #downsampled screen
        self.coords_pad = 12

        self.action_space = spaces.Discrete(len(self.valid_actions))

        self.observation_space = spaces.Dict(
            {
                "screens": spaces.Box(low=0, high=255, shape = self.output_shape, dtype = np.uint8),
                "health": spaces.Box(low=0, high=1),
                "level": spaces.Box(low=-1, high=1),
                "badges": spaces.MultiBinary(8),
                "map": spaces.Box(low=0, high=255, shape = (
                    self.coords_pad*4, self.coords_pad*4, 1), dtype = np.uint8),
                "recent_actions": spaces.MultiDiscrete([len(self.valid_actions)]*self.frame_stacks)
            }
        )

        head = "null" if config["headless"] else "SDL2"

        self.pyboy = pyboy.PyBoy(
            config["gb_path"],
            window = head
        )

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)

    def reset(self, seed=None, options = {}):
        self.seed = seed
        #restarts game, skipping credidts

        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)
        
        self.init_map_mem()
        
        self.agent_stats = []

        self.explore_map_dim = GLOBAL_MAP_SHAPE
        self.explore_map = torch.zeros(self.explore_map_dim, dtype = torch.uint8)

        self.recent_screens = torch.zeros(self.output_shape, dtype = torch.uint8)

        self.recent_actions = torch.zeros((self.frame_stacks), dtype = torch.uint8)

        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_level_reward = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        
        self.fitness = 0
        self.fitness_dict = {}

        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward() #TODO call differently, probably
        self.total_reward = 0
        self.reset_count +=1
        return self._get_obs(), {}

    def init_map_mem(self):
        self.seen_coords = {}

    def render(self, reduce_res = True):
        game_pixels_render = self.pyboy.screen.ndarray[:,:,0:1] #(144,160,3)
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2,2,1)).astype(np.uint8)
            )
        return torch.from_numpy(game_pixels_render)
    
    def _get_obs(self):

        screen = self.render()

        self.update_recent_screens(screen)

        level_sum = sum([self.read_m(a) for a in [0xD18c, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]])

        observation = {
            "screens": self.recent_screens,
            "health": torch.tensor([self.read_hp_fraction()]),
            "level": level_sum,
            "badges": torch.tensor([int(bit) for bit in f"{self.read_m(0xD356):08b}"], dtype = torch.int8),
            "map": self.get_explore_map()[:,:, None],
            "recent_actions": self.recent_actions 
        }    

        return observation

    def step(self, action):
        
        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        self.update_recent_actions(action)

        self.update_seen_coords()

        self.update_explore_map()

        self.update_heal_reward()

        self.party_size = self.read_m(0xD163)

        #self.fitness = self.update(fitness) #seems wrong to do adaptively, gotta track some other way, than evaluate at the end

        self.last_health = self.read_hp_fraction()

        self.update_map_progress()

        step_limit_reached = self.check_if_done()

        obs = self._get_obs()

        self.step_count += 1

        if step_limit_reached:
            self.fitness, self.fitness_dict = self.get_game_state_reward()

        return obs, self.fitness, step_limit_reached, self.fitness_dict

    def run_action_on_emulator(self,action):
        #press button
        self.pyboy.send_input(self.valid_actions[action])
        #render only if needed
        render_screen = not self.headless
        press_step = 8
        self.pyboy.tick(press_step, render_screen)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step -1, render_screen)
        self.pyboy.tick(1, True)

    def append_agent_stats(self, action):
        x_pos, y_pos, map_n = self.get_game_coords()
        levels = [
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD268]
        ]
        self.agent_stats.append(
            {
                "step": self.step_count,
                "x":x_pos,
                "y": y_pos,
                "map": map_n,
                "max_map_progress": self.max_map_progress,
                "last_action": action,
                "pcount": self.read_m(0xD163),
                "levels": levels,
                "levels_sum": sum(levels),
                "ptypes": self.read_party,
                "hp": self.read_hp_fraction(),
                "coord_count": len(self.seen_coords),
                "deaths": self.died_count,
                "badge": self.get_badges()
            }
        )
    
    def get_game_coords(self):
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))
    
    def update_seen_coords(self):
        #if not in batte
        if self.read_m(0xD057) == 0:
            x_pos, y_pos, map_n = self.get_game_coords()
            coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
            if coord_string in self.seen_coords.keys():
                self.seen_coords[coord_string] += 1
            else:
                self.seen_coords[coord_string] = 1

    def get_global_coords(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        return local_to_global(y_pos,x_pos,map_n)

    def update_explore_map(self):
        c = self.get_game_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            print(f"coord out of bounds! global: {c} game: {self.get_game_coords()}")
            pass
        else:
            self.explore_map[c[0], c[1]] = 255
    
    def get_explore_map(self):
        c = self.get_global_coords()
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            out = torch.zeros((self.coords_pad*2, self.coords_pad*2), dtype = torch.uint8)
        else:
            out = self.explore_map[
                c[0]-self.coords_pad:c[0] + self.coords_pad,
                c[1]-self.coords_pad:c[1] + self.coords_pad
            ]
        return repeat(out, 'h w -> (h h2) (w w2)', h2 = 2, w2=2) #TODO still need to figure out what is going on here

    def update_recent_screens(self, cur_screen):
        self.recent_screens = torch.roll(self.recent_screens, 1, 2)
        self.recent_screens[:,:,0] = cur_screen[:,:,0]

    def update_recent_actions(self, action):
        self.recent_actions = torch.roll(self.recent_actions, 1)
        self.recent_actions[0] = action
    
    def check_if_done(self):
        done = self.step_count >= self.max_steps -1
        return done

    def read_m(self, addr):
        return self.pyboy.memory[addr]
    
    def read_bit(self, addr, bit:int)-> bool:
        return bin(256+self.read_m(addr))[-bit -1] == "1"

    def get_levels_sum(self):
        min_poke_level = 2
        starter_additional_levels = 4
        poke_levels = [
            max(self.read_m(a) - min_poke_level, 0)
            for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        return max(sum(poke_levels) - starter_additional_levels, 0)
    
    def get_badges(self):
        return self.bit_count(self.read_m(0xD356))
    
    def read_party(self):
        return [
            self.read_m(addr)
            for addr in [0xD164, 0xD165, 0xD166, 0xD168, 0xD169]
        ]
    
    def update_max_op_level(self):
        opp_base_level = 5
        opponent_level = (
            max([
                self.read_m(a)
                for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]
            ])
            -opp_base_level
        )
        self.max_opponent_level = max(self.max_opponent_level, opponent_level)
        return self.max_opponent_level
    
    def update_heal_reward(self):
        cur_health = self.read_hp_fraction()
        if cur_health > self.last_health and self.read_m(0xD163) == self.party_size:
            if self.last_health > 0:
                pass
            else:
                self.died_count += 1
    
    def read_hp_fraction(self):
        hp_sum = sum([
            self.read_hp(add)
            for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
        ])
        max_hp_sum = sum([
            self.read_hp(add)
            for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
        ])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum/max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start +1)

    def bit_count(self, bits):
        return bin(bits).count("1")
    
    def update_map_progress(self):
        map_idx = self.read_m(0xD35E)
        self.max_map_progress = max(self.max_map_progress, self.get_map_progress(map_idx))
    
    def get_map_progress(self, map_idx):
        if map_idx in self.essential_map_locations.keys():
            return self.essential_map_locations[map_idx]
        else:
            return -1

    def get_current_coord_count_reward(self):
        x_pos, y_pos, map_n = self.get_game_coords()
        coord_string = f"x:{x_pos} y:{y_pos} m:{map_n}"
        if coord_string in self.seen_coords.keys():
            count = self.seen_coords[coord_string]
        else:
            count = 0
        
        if self.is_harsh:
            return 0 if count < self.stuck_threshold else count/10
        else:
            return 0 if count < self.stuck_threshold else 1      
    
    def get_game_state_reward(self, print_stats=False):
        state_scores= {
            "level": self.get_levels_sum() * 0.3,
            "heal": self.total_healing_rew * 0.1, #TODO
            "dead" : self.died_count * -0.1,
            "badge": self.get_badges()* 0.1,
            "explore": len(self.seen_coords)* 0.3, 
            "stuck": self.get_current_coord_count_reward() * -0.1 #not sure if useful
        }
        return sum(state_scores.values()), state_scores
