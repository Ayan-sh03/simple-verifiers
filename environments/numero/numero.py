import verifiers as vf
import random
from typing import Optional, List, Tuple, Dict


class NumeroEnv(vf.Environment):
    def __init__(self, low: int = 1, high: int = 50, max_turns: int = 20):
        self.low = low
        self.high = high
        self.max_turns = max_turns
        self.reset()

    def reset(self):
        """Start a new game and return initial observation."""
        self.secret = random.randint(self.low, self.high)
        self.turns = 0
        return {"low": self.low, "high": self.high}

    def step(self, action: int):
        """
        Take one step in the environment.

        action: the agent’s guess (an int).
        Returns: (observation, reward, done, info)
        """
        self.turns += 1

        observation = {"low": self.low, "high": self.high}

        if action == self.secret:
            return observation, 1.0, True, {"message": "Correct!"}
        elif self.turns >= self.max_turns:
            return observation, -1.0, True, {"message": f"Out of turns! The number was {self.secret}."}
        elif action < self.secret:
            return observation, -0.1, False, {"message": "Higher"}
        else:
            return observation, -0.1, False, {"message": "Lower"}

    def render(self):
        """Optional: human-readable output."""
        print(f"Guess a number between {self.low} and {self.high}")

    def rollout(self, policy_fn, max_steps: Optional[int] = None) -> List[Tuple[int, float, bool, Dict]]:
        """
        Runs a full episode using policy_fn, which maps a state to an action (an int guess).
        Returns a list of (action, reward, done, info) tuples.
        """
        history = []
        obs = self.reset()
        step = 0
        while True:
            action = policy_fn(obs)  # agent provides a guess based on obs
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
    """Prime hub entrypoint."""
    return NumeroEnv(**kwargs)
