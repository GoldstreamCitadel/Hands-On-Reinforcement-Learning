from policy_iter import CliffWalkingEnv, print_agent


class ValueIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0]*self.env.ncol*self.env.nrow
        self.theta = theta
        self.gamma = gamma
        # strategy we're looking for
        self.pi = [None for i in range(self.env.ncol*self.env.nrow)]

    def value_iter(self):
        cnt = 0
        while 1:
            max_diff = 0
            new_v = [0]*self.env.ncol*self.env.nrow
            for s in range(self.env.ncol*self.env.nrow):
                qsa_list=[]
                for a in range(4):
                    qsa=0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa+=p*(r+self.gamma*self.v[next_state]*(1-done))
                    qsa_list.append(qsa)
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s]-self.v[s]))
            self.v = new_v
            if max_diff<self.theta: break
            cnt += 1
        print("value-iter done after %d epoches."%cnt)
        self.get_policy()

    def get_policy(self):
        for s in range(self.env.nrow*self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa=0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p*(r+self.gamma*self.v[next_state]*(1-done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)
            self.pi[s]=[1/cntq if q==maxq else 0 for q in qsa_list]


if __name__ == '__main__':
    env = CliffWalkingEnv()
    action_meaning =['^','v','<','>']
    theta = 0.001
    gamma = 0.9
    agent = ValueIteration(env, theta, gamma)
    agent.value_iter()
    print_agent(agent, action_meaning, list(range(37, 47)),[47])