import gym
from linear_rl.custom_env import CustomEnv
import numpy as np
import matplotlib.pyplot as plt
from linear_rl.true_online_sarsa import TrueOnlineSarsaLambda
from linear_rl.fourier import FourierBasis
import sys
import copy
import math

# 移動平均を計算する関数
def moving_average(interval, window_size):
    if window_size == 1:
        return interval
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

#移動分散を計算する関数
def moving_variance(interval, window_size):
    if window_size == 1:
        return np.zeros_like(interval)  
    # 平均を計算するためのウィンドウ
    window = np.ones(int(window_size)) / float(window_size)
    
    # 平均 (移動平均) の計算
    moving_avg = np.convolve(interval, window, 'same')
    
    # 平方値の移動平均を計算
    squared_interval = np.array(interval)**2
    moving_avg_squared = np.convolve(squared_interval, window, 'same')
    
    # 分散を計算: 分散 = E[X^2] - (E[X])^2
    variance = moving_avg_squared - moving_avg**2
    
    return variance

#状態グリッドから最も近い状態を見つける    
def find_nearest_state(state_grid, position, velocity):

    # 位置の候補インデックスを取得
    position_distances = np.abs(state_grid[:, 0] - position)
    nearest_position_indices = np.where(position_distances == position_distances.min())[0]

    # 速度の候補インデックスを取得
    velocity_distances = np.abs(state_grid[nearest_position_indices, 1] - velocity)
    nearest_velocity_index = nearest_position_indices[np.argmin(velocity_distances)]

    return nearest_velocity_index

    
#θの計算(リッジ回帰)    
def compute_theta(p_next, D):
    return np.linalg.inv(D @ D.T + np.eye(D.shape[0], dtype=np.float32)) @ D @ p_next.reshape(-1, 1).astype(np.float32)



#⋂集合を計算する関数
def intersect_2d_with_set(grid1, grid2):
    # グリッドをタプル形式に変換（setで扱うため）
    set1 = set(map(tuple, grid1))
    set2 = set(map(tuple, grid2))
    
    # 共通集合を計算
    intersection = np.array(list(set1 & set2))
    return intersection

#確率的制御不変集合(PCIS)の計算
def compute_CIS(obs_history, state_grid, actions, xmin, xmax, vmin , vmax, Omega, N, epsilon, d, D, alpha_eta,agent):

   

    #グリッド数
    num_states=state_grid.shape[0]
    
     # p: 各状態の確率値
    p = np.zeros((num_states, 1))
    # p_tilda: 各時刻における確率値
    p_tilda = np.zeros((len(obs_history)-1  , 1))
    X_star_CIS = Omega


    while True:
        # 初期化: p_tilda を安全領域に基づいて設定

 
        for k in range(len(obs_history)-1):
            position = obs_history[k+1,0]
            velocity = obs_history[k+1,1]

            # 安全性を確認
            #if (xmin <= position <= xmax) and (vmin <= velocity <= vmax):
            #if (min(X_star_CIS[:,0]) <= position <= max(X_star_CIS[:,0])) and (min(X_star_CIS[:,1]) <= velocity <= max(X_star_CIS[:,1])):
            
            if np.any(np.all(np.isclose(X_star_CIS, [position, velocity], atol=1e-2), axis=1)):
                p_tilda[k, 0] = 1
            else:
                p_tilda[k, 0] = 0
            

        # 最終時刻の確率を 1 に設定
        p[:, 0] = 1
        

        # 再帰的に計算 (j = N から 1 まで)
        for j in range(N, 0, -1):
            # θ を計算 (リッジ推定)
            theta = compute_theta(p_tilda[:, 0], D)

    
            # 各状態について最大値を探索
            for grid_index, s in enumerate(state_grid):

                max_value = 0
                

                phi_xu = agent.get_features(s).reshape(d, 1)

                for u_index, u in enumerate(actions):
                    for a in range(agent.action_dim):
                        if a == u:
                            expected_value = (
                                theta.T @ phi_xu
                                - alpha_eta
                                * np.sqrt(phi_xu.T @ np.linalg.inv(D @ D.T + np.eye(D.shape[0])) @ phi_xu)
                            )
                   
                                
                        else:
                            expected_value = 0
         
                        max_value = max(max_value, max(0, expected_value))
                    
                # 状態 x の確率を更新
                p[grid_index, 0] = p[grid_index, 0] * max_value

     
            # p_tilda を更新
            for k in range(len(obs_history) - 1):
                index=find_nearest_state(state_grid,obs_history[k,0],obs_history[k,1])
                 
                p_tilda[k, 0] = p[index, 0]


        print("pmin",min(p[:,0]))
        print("pmax",max(p[:,0]))
        # 新しい制御不変集合を計算
        indices = np.where(p[:, 0] >= 1 - epsilon)[0]

        X_star_CIS_new = state_grid[indices]

        print(len(X_star_CIS))
        print(len(X_star_CIS_new))
       

        set_X_star_CIS = set(map(tuple, X_star_CIS))
        set_X_star_CIS_new = set(map(tuple, X_star_CIS_new))

        # 空集合のときはゼロ除算防止
        if len(set_X_star_CIS | set_X_star_CIS_new) == 0:
            jaccard = 1.0
        else:
            jaccard = len(set_X_star_CIS & set_X_star_CIS_new) / len(set_X_star_CIS | set_X_star_CIS_new)
        
        print("jaccard",jaccard)

        if jaccard > 0.95:
            break

        """
        if set_X_star_CIS == set_X_star_CIS_new:
            break
        """

        X_star_CIS = intersect_2d_with_set(X_star_CIS, X_star_CIS_new)
        
        
        print(min(X_star_CIS[:,0]))
        print(max(X_star_CIS[:,0]))
        print(min(X_star_CIS[:,1]))
        print(max(X_star_CIS[:,1]))


    return X_star_CIS, p

#状態遷移関数
def custom_state_transition(current_state, action):

    force=0.001
    gravity=0.0025
    #max_speed=0.07

    #min_position=-1.5
    #max_position=0.6
    position, velocity = current_state

    # 行動に応じた加速度を計算
    velocity += (action - 1) * force - gravity * math.cos(3 * position)
    #velocity = np.clip(velocity, -max_speed, max_speed)

    # 位置を更新
    position += velocity
    #position = np.clip(position, min_position, max_position)

    # 左端で速度をリセット
    #if position == min_position and velocity < 0:
    #    velocity = 0

    next_state = np.array([position, velocity], dtype=np.float32)

    return next_state




def main():
    # 違反回数を記録するリスト
    violations_iteration_0 = []
    violations_iteration_1 = []

    

    rets_0=[]
    rets_1=[]


    epochs_0=[]
    epochs_1=[]

    #iteration=0:提案手法
    #iteration=1:既存手法
    for iteration in range(2):
        
        env = CustomEnv()
        fourier_order = 5
        max_non_zero = 2
        
        basis = FourierBasis(env.observation_space, env.action_space, fourier_order, max_non_zero)
        agent = TrueOnlineSarsaLambda(env.observation_space, env.action_space,
                                    alpha=0.001, fourier_order=fourier_order, gamma=0.99,
                                    lamb=0.9, epsilon=0.5, min_max_norm=True)

        
        #環境をリセットして初期観測を取得
        obs = env.reset()
        

        ret = 0
        rets = []
        episodes = 100
        ep = 0
        all_positions = []
        time_steps = []
        cumulative_violations = []
        total_violations = 0
        constraint_value = 1

        CIS_map_data = []

        epochs=[]
        

        # 安全領域（例: 位置 -1.5 から 0.6）
        xmin = -1.5
        xmax = 0.6
        vmin = -0.07
        vmax = 0.07
        

        position_range = (xmin,xmax)
        velocity_range = (vmin,vmax)
        num_position = 200
        num_velocity = 30
        position_values = np.linspace(position_range[0], position_range[1], num_position)
        velocity_values = np.linspace(velocity_range[0], velocity_range[1], num_velocity)
        position_grid, velocity_grid = np.meshgrid(position_values, velocity_values)
        state_grid = np.column_stack((position_grid.ravel(), velocity_grid.ravel()))
        

        grid_index=2
        print(state_grid[grid_index])#state_grid[0]からstate_grid[499]まで
        position = state_grid[grid_index, 0]  # Position 値を取得
        velocity = state_grid[grid_index, 1]  # Velocity 値を取得
        print(f"Position = {position}, Velocity = {velocity}")
        print(f"State grid shape: {state_grid.shape}")


        #安全領域PCISを設定

        #集合Ω_0
        safe_position_range = (xmin,xmax)
        safe_velocity_range = (vmin, vmax)
        safe_num_position = 200 #positionの分割数
        safe_num_velocity = 30  #velocityの分割数
        safe_position_values = np.linspace(safe_position_range[0], safe_position_range[1], safe_num_position)
        safe_velocity_values = np.linspace(safe_velocity_range[0], safe_velocity_range[1], safe_num_velocity)
        safe_position_grid, safe_velocity_grid = np.meshgrid(safe_position_values, safe_velocity_values)
        safe_state_grid = np.column_stack((safe_position_grid.ravel(), safe_velocity_grid.ravel()))

        Omega = safe_state_grid

        #集合を計算する時のパラメータ
        epsilon = 0.3
        alpha_eta = 0.1

        #計画行列Dの初期化
        D = np.array([])

        d = basis.get_num_basis()
        #print("基底関数の次元",d)

        

        #行動集合[0,1,2]
        actions = list(range(3))

        #安全な行動集合
        safeact =actions

        time_steps=[]
        



        all_positions.append(obs[0])

        time_steps.append(len(all_positions))

        obs_history=[]
        #初期状態を保存
        obs_history.append(obs)

       
        #ε-greedy方策のεを設定
        initial_epsilon = 0.5  # 初期値
        final_epsilon = 0.01    # 最小値
        decay_rate = 0.05      # エピソードごとの減少率

        #1回のエピソードで生成する状態数
        max_step=300
        step_count=0

        done = False
        
    
        while ep < episodes:

            #ε-greedy方策のεをエピソードごとに減らす
            agent.epsilon = max(final_epsilon, initial_epsilon - decay_rate * ep)
            



            #iteration==0：提案手法
            if iteration==0:
                    if ep==0:                        
                        action=agent.act(obs,safeact)
                        new_obs, rew, done, info = env.step(action)  
                             
                    else:
                        
                        safeact=[]
                        #次の状態が確率的制御不変集合内に入る安全な行動集合safeactを構築
                        for action in actions:
                            # 特徴量を計算
                            features = agent.get_features(obs)
                            theta = compute_theta(np.array(CIS_map_data), D_pre)
                                
                            # 分散を計算
                            variance = features.T @ np.linalg.inv(D_pre @ D_pre.T + np.eye(D_pre.shape[0])) @ features
                                
                            # 制約値を計算
                            value = theta.T @ features - alpha_eta * np.sqrt(variance)
                        
                            
                        
                            #安全な行動の場合safeactに格納
                            if value > 0: 
                                safeact.append(action)
                            """
                            next_state=custom_state_transition(obs,action)
                            nearest_position_idx = np.argmin(np.abs(position_values - next_state[0]))
                            nearest_velocity_idx = np.argmin(np.abs(velocity_values - next_state[1]))
                            valid_value = valid_region[nearest_velocity_idx, nearest_position_idx]

                            #安全領域から外れた場合
                            if xmin>next_state[0] or xmax<next_state[0] or vmin>next_state[1] or vmax<next_state[1]:
                                valid_value=0
                            
                            #安全な行動の場合safeactに格納
                            if valid_value :
                                safeact.append(action)
                            
                            """

                            

                        
               


                            


                            
                            
                        if not safeact: # safeact が空リストなら
                            print("\n[ERROR] 安全な制御入力が見つかりませんでした。シミュレーションを終了します。")
                            print(f"エピソード: {ep}, 状態: {obs}, 許容可能なアクション: {safeact}")
                            
                            # PCIS と制約値の可視化
                            plt.figure(figsize=(10, 8))
                            plt.contourf(position_grid, velocity_grid, valid_region, levels=[0, 0.5, 1], colors=["white", "green"], alpha=0.7)
                            plt.colorbar(label="Constraint Validity (1: Valid, 0: Invalid)")

                            # 軌道の可視化
                            obs_positions = np.array([s[0] for s in obs_history])
                            obs_velocities = np.array([s[1] for s in obs_history])
                            plt.plot(obs_positions, obs_velocities, color="blue", linewidth=2, label="Trajectory (obs_history)")
                            plt.scatter(obs_positions, obs_velocities, color="blue", label="Trajectory (Points)", zorder=5)

                            plt.xlabel("Position")
                            plt.ylabel("Velocity")
                            plt.title("Valid Region for Constraint Value & Trajectory")
                            plt.grid(True)
                            plt.legend()
                            plt.show()

                            sys.exit(1)  # 異常終了

                        #安全な行動集合safeactの中からε-greedy方策に基づき行動選択
                        action=agent.act(obs,safeact)

                        #状態遷移
                        new_obs, rew, done, info = env.step(action)
                        


            #iteration==1：既存手法
                 
            elif iteration ==1:
                #行動集合actionsの中からε-greedy方策に基づき行動選択
                action=agent.act(obs,actions)

                #状態遷移
                new_obs, rew, done, info = env.step(action)
            
           
            obs_history.append(new_obs)

            # 違反チェック
            position = new_obs[0]
         
            velocity = new_obs[1]

            
                

            if not ((xmin <= position <= xmax) and (vmin <= velocity <= vmax)):
                
                total_violations += 1
            cumulative_violations.append(total_violations)
            epochs.append(ep)
 

            # 学習処理
            ret += rew
            agent.learn(obs, action,safeact, rew, new_obs, done)
            obs = new_obs
            step_count+=1
    
            all_positions.append(position)

            #ステップ数を計算
            time_steps.append(len(all_positions))

   
        

            features = agent.get_features(obs).reshape(d, 1)
            if D.size == 0:
                D = features
            else:
                D = np.hstack([D, features])
            
            if done or step_count >= max_step:
                step_count = 0
                print("ep",ep)
              
                obs_history_array = np.array(obs_history)
                
                print("obs_history[0]",obs_history_array[0])
                
                print("positionsの最小",min(obs_history_array[:,0]))
                print("positionsの最大",max(obs_history_array[:,0]))
            
                print("velocityの最小",min(obs_history_array[:,1]))
                print("velocityの最大",max(obs_history_array[:,1]))
                
                #確率的制御不変集合の計算
                X_star_CIS, p = compute_CIS(obs_history_array, state_grid, actions, xmin, xmax,vmin,vmax, Omega, 1, epsilon,  d, D, alpha_eta ,agent)
               

                CIS_map_data=[]
                D_pre=D


                #確率的制御不変集合内に入る状態軌道を1、集合の外に出る状態を0としてグリッドをマッピング
                for k in range(len(obs_history_array) - 1):
                    position = obs_history_array[k+1,0]
                    velocity = obs_history_array[k+1,1]

                    if (np.min(X_star_CIS[:,0]) <= position <= np.max(X_star_CIS[:,0])) and (np.min(X_star_CIS[:,1]) <= velocity <= np.max(X_star_CIS[:,1])):
                 
                        CIS_map_data.append(1)
                    else:
                        CIS_map_data.append(0)
          

                    
                # constraint_value を格納するための配列
                constraint_map = np.zeros_like(position_grid)

                for i in range(position_grid.shape[0]):
                    for j in range(position_grid.shape[1]):
                        # 現在の位置と速度
                        position = position_grid[i, j]
                        velocity = velocity_grid[i, j]
                        obs_grid = np.array([position, velocity])  # 状態
                            
                        # 特徴量を計算
                        features = agent.get_features(obs_grid)
                        theta = compute_theta(np.array(CIS_map_data), D)
                            
                        # 分散を計算
                        variance = features.T @ np.linalg.inv(D @ D.T + np.eye(D.shape[0])) @ features
                            
                        # 制約値を計算
                        constraint_value = theta.T @ features - alpha_eta * np.sqrt(variance)
                            
                        # constraint_value をマップに格納
                        constraint_map[i, j] = constraint_value


                # constraint_value > 0 の範囲をマスク
                valid_region = constraint_map > 0

                              
                
                # 状態軌道のプロット
                plt.figure(figsize=(10, 8))
                if ep!=0:
                    if iteration==0:
                        plt.contourf(position_grid, velocity_grid, pre_valid_region, levels=[0, 0.5, 1], colors=["white", "green"], alpha=0.7)
                        plt.colorbar(label="Constraint Validity (1: Valid, 0: Invalid)")
                        print("PCIS最小位置",np.min(valid_positions))
                        print("PCIS最大位置",np.max(valid_positions))
                        print("PCIS最小速度",np.min(valid_velocities))
                        print("PCIS最大速度",np.max(valid_velocities))


                obs_positions = obs_history_array[:, 0]
                obs_velocities = obs_history_array[:, 1]
                plt.plot(obs_positions, obs_velocities, color="blue", linewidth=2, label="Trajectory (obs_history)")
                # 点をプロット
                plt.scatter(obs_positions, obs_velocities, color="blue", label="Trajectory (Points)", zorder=5)

            
                x_min,x_max=xmin,xmax
                y_min,y_max=vmin,vmax
                x_margin = 0.1 * (x_max - x_min)  
                y_margin = 0.1 * (y_max - y_min)  
                plt.xlim(x_min - x_margin, x_max + x_margin)
                plt.ylim(y_min - y_margin, y_max + y_margin)
                plt.xlabel("Position")
                plt.ylabel("Velocity")
                plt.title("Valid Region for Constraint Value")
                plt.grid(True)
                plt.show()

                
                pre_valid_region = valid_region
                # 有効領域 (valid_region) に対応する位置と速度を取得
                valid_positions = position_grid[pre_valid_region]
                valid_velocities = velocity_grid[pre_valid_region]


                rets.append(ret)
                ret = 0
                ep += 1
              
                

            if done:
                print("目標位置到達")
                break

        
                

        # データの保存
        if iteration == 0:
            violations_iteration_0 = cumulative_violations[:]            
            proposal_positions = obs_history_array[:, 0]
            proposal_velocities = obs_history_array[:, 1]
            proposal_valid_region = valid_region
            rets_0 = rets[:]           
            epochs_0=epochs[:]

        elif iteration == 1:
            violations_iteration_1 = cumulative_violations[:]
            existing_positions = obs_history_array[:, 0]
            existing_velocities = obs_history_array[:, 1]
            rets_1 = rets[:]
            epochs_1 = epochs[:]

    # デバッグ用のログ
    print("Violations Iteration 0:", len(violations_iteration_0))
    print("epoch 0:", len(epochs_0))
    
    
    print("Violations Iteration 1:", len(violations_iteration_1))
    print("epoch 1:", len(epochs_1))

    # === 提案手法のグラフ ===
    plt.figure(figsize=(10, 8))

    # 緑の領域 (PCIS)
    plt.contourf(position_grid, velocity_grid, proposal_valid_region, levels=[0, 0.5, 1], colors=["white", "green"], alpha=0.7, label="Valid Region (Proposal)")

    # 状態軌道
    plt.plot(proposal_positions, proposal_velocities, color="blue", linewidth=2,  label="Proposal Trajectory")
    plt.scatter(proposal_positions, proposal_velocities, color="blue", zorder=5)
    x_min,x_max=xmin,xmax
    y_min,y_max=vmin,vmax
    x_margin = 0.1 * (x_max - x_min)  
    y_margin = 0.1 * (y_max - y_min)  
    plt.xlim(x_min - x_margin, x_max + x_margin)
    plt.ylim(y_min - y_margin, y_max + y_margin)

    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title("Proposal Method: Valid Region & Trajectory")
    plt.legend()
    plt.grid(True)
   

    # === 既存手法のグラフ ===
    plt.figure(figsize=(10, 8))

    # 状態軌道
    plt.plot(existing_positions, existing_velocities, color="blue", linewidth=2, label="Existing Method Trajectory")
    plt.scatter(existing_positions, existing_velocities, color="blue", zorder=5)

    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title("Existing Method: Trajectory")
    plt.legend()
    plt.grid(True)




    # 違反回数のプロット (エポックごと)
    plt.figure(figsize=(10, 6))  # 新しい図を作成
    plt.plot(epochs_1, violations_iteration_1, label="existing methods", color="blue")
    plt.plot(epochs_0, violations_iteration_0, label="Proposed Methods", color="red")
    plt.xlabel("Episodes")  # 横軸をエポックに
    plt.ylabel("Violations")
    plt.title("Comparison of Violations")
    plt.legend()
    plt.grid(True)
    

   

    #報酬
    plt.figure(figsize=(10, 6))
    plt.plot(moving_average(rets_1, 1),label="existing methods",color="blue")
    plt.plot(moving_average(rets_0, 1),label="Proposed Methods",color="red")
    plt.xlabel("Episodes")
    plt.ylabel("cumulative reward")
    plt.title("cumulative reward per Episode")
    plt.legend()  # 凡例を表示する
    plt.grid(True)  # グリッドを追加して見やすくする
    



    # 平均と分散を比較する棒グラフを作成
    plt.figure(figsize=(10, 6))
    plt.plot(moving_variance(rets_1, 5),label="existing methods",color="blue")
    #print("rets_0_var",moving_variance(rets_0, 2))
    plt.plot(moving_variance(rets_0, 5),label="Proposed Methods",color="red")
    plt.xlabel("Episodes")
    plt.ylabel("Return Variance")
    plt.title("Return variance per Episode")
    plt.legend()
    plt.grid(True)
    plt.show()
    



if __name__ == '__main__':
    main()


