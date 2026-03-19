from trainer import DynaQ_Trainer
import numpy as np
import random
import time
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']

if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    n_planning_list = [0, 2, 20]
    for n_planning in n_planning_list:
        print('>>> Q-planning steps:%d' % n_planning)
        time.sleep(0.5)
        return_list = DynaQ_Trainer(n_planning)
        episodes_list = list(range(len(return_list)))
        plt.plot(
            episodes_list,
            return_list,
            label=str(n_planning)+' planning steps'
        )
    # 这块儿注意缩进，全训完了再画，对比
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Dyna-Q on {}'.format('冰窟窿\n所以您看呐，这复盘越多，它收敛得就越快'))
    plt.show()