import verifiers as vf
import random
from math import exp
from typing import Tuple, Optional, Dict, List

class Caliente(vf.Environment):
    def __init__(self, row: int, col: int, max_turns: Optional[int] = None):
        self.row = row
        self.col = col
        self.max_turns = max_turns or (row * col)
        self.reset()

    def reset(self) -> Dict:
        self.x = random.randint(0, self.row - 1)
        self.y = random.randint(0, self.col - 1)
        self.turns = 0
        return {"row": self.row, "col": self.col}

    @staticmethod
    def score(dx: int, dy: int) -> float:
        d = abs(dx) + abs(dy)
        return 1.0 if d == 0 else exp(-d / 2.5)

    def step(self, action: Tuple[int, int]):
        ax, ay = action
        if not (0 <= ax < self.row and 0 <= ay < self.col):
            return {"row": self.row, "col": self.col}, -1.0, False, {"message": "out_of_bounds"}

        self.turns += 1
        dx, dy = ax - self.x, ay - self.y
        distance = abs(dx) + abs(dy)
        done = (distance == 0) or (self.turns >= self.max_turns)
        reward = 1.0 if distance == 0 else (self.score(dx, dy) if not done else -1.0)

        if distance == 0:
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

        obs = {"row": self.row, "col": self.col}
        return obs, reward, done, {"message": msg, "distance": distance}

def load_environment(**kwargs) -> vf.Environment:
    return Caliente(**kwargs)