import verifiers as vf
import random
from typing import Tuple,Optional,Dict,List
from math import exp


class Caliente(vf.Environment):
    def __init__(self,row:int , col:int, max_turns:Optional[int]):
        self.row = row
        self.col = col
        self.secret : Tuple[int,int] = (row,col)
        self.max_turns  = max_turns or row*col # defaults to total elements , can be changed for stricter constraints

        self.reset()


    def reset(self):
        """Start a new game with fresh board and return initial setup"""
        self.x = random.randint(0, self.row - 1)
        self.y = random.randint(0, self.col - 1)
        self.turns = 0
        return {"row": self.row, "col": self.col}
    
    def score(dx: int, dy: int) -> float:
        d = abs(dx) + abs(dy)
        if d == 0:
            return 1.0
        return exp(-d / 2.5)   # 2.5 keeps ~0.2 at d=4

    def step(self, action: Tuple[int, int]):
        self.turns += 1
        dx = action[0] - self.x
        dy = action[1] - self.y
        distance = abs(dx) + abs(dy)

        obs = {"row": self.row, "col": self.col}
        done = (distance == 0)
        
        reward = 1.0 if done else exp(-distance / 2.5)

        # human-readable hint
        if done:
            msg = "Correct!"
        elif distance == 1:
            msg = "burning"
        elif distance == 2:
            msg = "hot"
        elif distance <= 4:
            msg = "warm"
        elif distance <= 6:
            msg = "cool"
        else:
            msg = "cold"

        info = {"message": msg, "distance": distance}
        return obs, reward, done, info
    
    
    def render(self):
        print(f"Guess a coordinate between (0, 0) and ({self.row - 1}, {self.col - 1})")

    def rollout(self, policy_fn, max_steps: Optional[int] = None) -> List[Tuple[Tuple[int, int], float, bool, Dict]]:
        
        history = []
        obs = self.reset()
        
        step = 0
        while True:
            action = policy_fn(obs)  # agent provides a guess based on observations
            obs_next, reward, done, info = self.step(action)
            history.append((action, reward, done, info))
            
            if done:
                break
            obs = obs_next
            step += 1
            if max_steps and step >= max_steps:
                break
            
        return history
        
def load_environment(**kwargs) -> vf.Environment:
    '''
    Loads a custom environment.
    '''
    return Caliente(**kwargs)
