class CliffWalkingEnv:
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        # P[state][action] = [(p, next_state, reward, done)]
        self.P = self.createP()

    def createP(self):
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        change = [[0,-1],[0,1],[-1,0],[1,0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # side
                    if i == self.nrow -1 and j>0:
                        P[i*self.ncol+j][a]=[
                            (1, i*self.ncol+j, 0, True)
                        ]
                        continue
                    
                    next_x = min(self.ncol-1, max(0,j+change[a][0]))
                    next_y = min(self.nrow-1, max(0,i+change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False

                    # next side or terminal
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol-1: # next not terminal, but side.
                            reward = -100 

                    P[i*self.ncol+j][a]=[(1,next_state,reward,done)]
                
        return P
    

class PolicyIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow
        self.pi = [[0.25, 0.25, 0.25, 0.25]
                   for i in range(self.env.ncol * self.env.nrow)]
        
        self.theta = theta
        self.gamma = gamma

    def policy_evaluation(self):
        cnt = 1
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma*self.v[next_state]*(1-done))
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s]-self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break
            cnt += 1
        print("strategy evaluation finished after %d epoches."%cnt)

    def policy_improvement(self):
        pass

    def policy_iteration(self):
        pass