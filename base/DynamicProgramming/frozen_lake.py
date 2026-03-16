import gym #from openai
#seems the lake has changed
env = gym.make("FrozenLake-v1")
env = env.unwrapped
env.render()

holes = set()
ends = set()

for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2] == 1.0:
                ends.add(s_[1])
            if s_[3] == True:
                holes.add(s_[1])
holes = holes - ends
print("Interuption holes idx:",holes)
print("Targets idx:",holes)

for a in env.P[14]:
    print(env.P[14][a])