# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 11:17:05 2022

@author: Bhavesh Kilaru, Naveen Kumar Gorantla
"""

import matplotlib.pylab as plt
import ReinforcementLearning as TicTacToeEnv
import timeit
def Q_value_convergencePlot():
    
    start_q = timeit.default_timer()
    game = TicTacToeEnv.playTicTacToe('q', 100000, alpha = 0.5)
    game.trainSmartAgent()
    stop_q = timeit.default_timer()
    Q_learning = game.q_value
    
    start_s = timeit.default_timer()
    game = TicTacToeEnv.playTicTacToe('s', 100000, alpha = 0.5)
    game.trainSmartAgent()
    stop_s = timeit.default_timer()
    sarsa = game.q_value
    
    start_sl = timeit.default_timer()
    game = TicTacToeEnv.playTicTacToe('sl',40000, alpha = 0.5)
    game.trainSmartAgent()
    stop_sl = timeit.default_timer()
    s_l = game.q_value
    
    plt.plot(Q_learning, label = 'Q learning')
    plt.plot(sarsa, label = "SARSA")
    plt.plot(s_l, label = "Sarsa(lambda)")
    plt.title("# of episodes vs Q value for alpha 0.5")
    plt.xlabel("# of episodes")
    plt.ylabel("Q value")
    plt.legend()
    plt.show()
    
    '''printing the execution times'''
    print('Q learning Time of learning for 500000 episodes: ', stop_q - start_q)  
    print('Sarsa Time of learning for 500000 episodes: ', stop_s - start_s)  
    print('Sarsa lambda Time of learning for 40000 episodes: ', stop_sl - start_sl)  
    


def plot_Q_Value_multiAlpha():
    
    gameq9 = TicTacToeEnv.playTicTacToe('q', 400000, alpha = 0.9)
    gameq9.trainSmartAgent()
    Q_learning_9 = gameq9.q_value
   
    gameq5 = TicTacToeEnv.playTicTacToe('q', 400000, alpha = 0.5)
    gameq5.trainSmartAgent()
    Q_learning_5 = gameq5.q_value
    gameq1 = TicTacToeEnv.playTicTacToe('q', 400000, alpha = 0.1)
    gameq1.trainSmartAgent()
    Q_learning_1 = gameq1.q_value
    
    plt.plot(Q_learning_9, label = 'Q learning for alpha 0.9')
    plt.plot(Q_learning_5, label = 'Q learning for alpha 0.5')
    plt.plot(Q_learning_1, label = 'Q learning for alpha 0.1') 
    
    plt.title("# of episodes vs Q value for differnt values of alpha")
    plt.xlabel("# of episodes")
    plt.ylabel("Q value")
    plt.legend()
    plt.show()

def discounted_cumulative_rewardPlot():
    game_q = TicTacToeEnv.playTicTacToe('q', 100000)
    game_q.trainSmartAgent()
    r1 = game_q.agent.rewards
    game_s = TicTacToeEnv.playTicTacToe('s', 100000)
    game_s.trainSmartAgent()
    r2 = game_s.agent.rewards
    game_sl = TicTacToeEnv.playTicTacToe('sl', 40000)
    game_sl.trainSmartAgent()
    r3 = game_sl.agent.rewards
    list1 = [r1 ,r2, r3]
    TicTacToeEnv.plotGraph_discountedReward(list1)
    
if __name__ == "__main__":
    print("select A for Q value value convergence plot for different learning with fixed alpha value")
    print("select B for Q value value convergence plot with multi alpha values")
    print("select C for discounted Cumulative reward ")
    plotType = input("Please choose plot type: ")
    if plotType.upper() == 'A':
        Q_value_convergencePlot()
    elif plotType.upper() == 'B':
        plot_Q_Value_multiAlpha()
    elif plotType.upper() == 'C':
        discounted_cumulative_rewardPlot()
    else:
        print("invalid selection")
        
    