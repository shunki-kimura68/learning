import numpy as np
import matplotlib.pyplot as plt


def transition_probability(S, A, states, actions, xmin, xmax):
    P = np.zeros((S + 1, A, S + 1))
    for i in range(S):
        x = states[i]
        for j in range(A):
            a = actions[j]
            next_index = f(x, a, states, xmin, xmax)
            if next_index != 0:
                P[i, j, next_index] = 0.9
                if 1 < next_index < S:
                    P[i, j, next_index + 1] = 0.05
                    P[i, j, next_index - 1] = 0.05
                elif next_index == 1:
                    P[i, j, next_index + 1] = 0.1
                elif next_index == S:
                    P[i, j, next_index - 1] = 0.1
            else:
                P[i, j, S] = 1.0
    for j in range(A):
        P[S, j, S] = 1
    return P


def compute_next_state(x, u):
    return x + 0.1 * (-0.2 * x + (1 - u) * np.sin(0.1 * (x + 10)))


def f(x, u, states, xmin, xmax):
    x_next = compute_next_state(x, u)
    if x_next < xmin or x_next > xmax:
        return 0
    return np.argmin(np.abs(states - x_next))

def curindex(x, states, xmin, xmax):
    if x < xmin or x > xmax:
        return 0
    else:
        return np.argmin(np.abs(states - x))



def compute_theta(p_next, D):
    
    #print(f"Dの行数: {D.shape[0]}, 列数: {D.shape[1]}")  # D のサイズを表示
    #print(f"p_nextの行数: {p_next.shape[0]}, 列数: {p_next.shape[1]}")  # p_next のサイズを表示
    #return np.linalg.inv(D @ D.T + np.eye(D.shape[0])) @ D @ p_next.reshape(-1,1)
    return np.linalg.inv(D @ D.T + np.eye(D.shape[0], dtype=np.float32)) @ D @ p_next.reshape(-1, 1).astype(np.float32)




def phi(x, u, states, actions, Phi, d):
    x_index = np.argmin(np.abs(states - x))
    u_index = np.argmin(np.abs(actions - u))
    return Phi[x_index, u_index, :].reshape(d, 1)


def compute_CIS(x_trajectory, states, actions, xmin, xmax, Omega, N, epsilon, Phi, d, D, alpha_eta,iteration):
    # p: 各状態の確率値
    p = np.zeros((len(states), 1))
    # p_tilda: 各時刻における確率値
    p_tilda = np.zeros((len(x_trajectory) - 1*iteration, 1))
    X_star_CIS = Omega

    while True:
        # 初期化: p_tilda を安全領域に基づいて設定
        for k in range(len(x_trajectory) - 1*iteration):
            if min(X_star_CIS) <= x_trajectory[k + 1] <= max(X_star_CIS):
                p_tilda[k, 0] = 1
            else:
                p_tilda[k, 0] = 0

        #print(f"p_tilda の初期行数: {p_tilda.shape[0]}, 列数: {p_tilda.shape[1]}")
        #print("p_tilda (初期化):", p_tilda)
        # 最終時刻の確率を 1 に設定
        p[:, 0] = 1

    
        for j in range(N, 0, -1):  # 再帰的に計算 (j = N から 1 まで)
            # θ を計算 (リッジ推定)
            theta = compute_theta(p_tilda[:, 0], D)
            #print(f"theta (ステップ {j}): {theta.flatten()}")

            # 各状態について最大値を探索
            for x_index, x in enumerate(states):
                max_value = 0
                for u_index, u in enumerate(actions):
                    # φ(x, u) を計算
                    phi_xu = phi(x, u, states, actions, Phi, d)
                    # 確率期待値を計算
                    
                    expected_value = (
                        theta.T @ phi_xu
                        - alpha_eta
                        * np.sqrt(phi_xu.T @ np.linalg.inv(D @ D.T + np.eye(D.shape[0])) @ phi_xu)
                    )
                    
                    max_value = max(max_value, max(0, expected_value.squeeze()))
                # 状態 x の確率を更新
                
                p[x_index, 0] = p[x_index, 0] * max_value

            # p_tilda を更新
            for k in range(len(x_trajectory) - 1*iteration):
                index = curindex(x_trajectory[k + 1], states, xmin, xmax)
                p_tilda[k, 0] = p[index, 0]
   

        # 新しい制御不変集合を計算
        #print("p",p[:,0])
        indices = np.where(p[:, 0] >= 1 - epsilon)[0]
        X_star_CIS_new = states[indices]

        # 収束条件を判定
        if np.array_equal(X_star_CIS, np.intersect1d(X_star_CIS, X_star_CIS_new)):
            break

        # 集合を更新
        X_star_CIS = np.intersect1d(X_star_CIS, X_star_CIS_new)

    return X_star_CIS, p


def select_control_input_for_max_variance(x, CISmap_data, states, actions, Phi, d, D, alpha_eta, X_star_CIS):
    curind = curindex(x, states, states.min(), states.max())
    max_variance = -np.inf
    best_u_index = None
    u_flag = 0
    
    # 制御入力を探索
    for i, u in enumerate(actions):
        phi_x_u = Phi[curind, i, :].reshape(d, 1)
        variance = phi_x_u.T @ np.linalg.inv(D @ D.T + np.eye(D.shape[0])) @ phi_x_u
        #print(f"CISmap_data の要素数: {CISmap_data.shape[0]}")
        theta = compute_theta(CISmap_data, D)
        constraint_value = theta.T @ phi_x_u - alpha_eta*np.sqrt(variance)
        
        #print(constraint_value.squeeze())
        #print("thetaphi",theta.T@phi_x_u)
        #print("√variance",np.sqrt(variance))
        # 安全性と分散を考慮
        
        if constraint_value.squeeze()> 0 :
            u_flag = 1
            if constraint_value > max_variance:
                max_variance = constraint_value
                best_u_index = i


    if np.random.rand() > 0.5:
        barflag=0
        while barflag==0:
            i=np.random.randint(len(actions))
            phi_x_u = Phi[curind, i, :].reshape(d, 1)
            variance = phi_x_u.T @ np.linalg.inv(D @ D.T + np.eye(D.shape[0])) @ phi_x_u
            theta = compute_theta(CISmap_data, D)
            constraint_value = theta.T @ phi_x_u - alpha_eta*np.sqrt(variance)
            if constraint_value.squeeze()> 0 :
                best_u_index = i
                barflag=1



    #print("θTΦ-√ΦT(DTD+I)Φ",max_variance)






    #print(best_u_index)
    return u_flag, best_u_index




def generate_initial_trajectory(x0, T, states, actions, Phi, D,d):
    x_trajectory = np.zeros(T)
    x_trajectory[0] = x0

    for k in range(T - 1):
        curind = curindex(x_trajectory[k], states, states.min(), states.max())

        # Randomly select a control input
        u = np.random.choice(actions)

        min_region, max_region = -0.1, 0.1
        # Check if the next state lies within the initial safety region
        x_next = compute_next_state(x_trajectory[k], u)
        if x_next < min_region or x_next > max_region:
            x_next = x_trajectory[k]  # Stay in the current state if unsafe

        # Update trajectory and D matrix
        #phidata = Phi[curind, np.argmin(np.abs(actions - u)), :].reshape(d, 1)
        phidata = Phi[curind, np.argmin(np.abs(actions - u)), :].reshape(d, 1).astype(np.float32)
        D = np.hstack([D, phidata]) if D.size else phidata

        
        

        x_trajectory[k + 1] = x_next


    return x_trajectory, D

def generate_trajectory(x0, T, states, actions, Phi, D, X_star_CIS, d, CISmap_data, alpha_eta):
    x_trajectory = np.zeros(T)
    x_trajectory[0] = x0
    

    for k in range(T - 1):
        curind = curindex(x_trajectory[k], states, states.min(), states.max())
        if k==100 or k==200 or k==300 or k==400 or k==500 or k==600 or k==700 or k==800 or k==900 :
            print(k)

        # 制御入力を選択
        u_flag, best_u_index = select_control_input_for_max_variance(x_trajectory[k], CISmap_data, states, actions, Phi, d, D, alpha_eta, X_star_CIS)


        if u_flag==0:
            print("安全な制御入力が見つかりません")
        u = actions[best_u_index]
        #print("u",u)
        
        # Phi データを更新
        #phidata = Phi[curind, best_u_index, :].reshape(d, 1)
        phidata = Phi[curind, best_u_index, :].reshape(d, 1).astype(np.float32)
        D = np.hstack([D, phidata]) 
        #print(f"Dの行数: {D.shape[0]}, 列数: {D.shape[1]}")

        # 次の状態を計算
        next_state = compute_next_state(x_trajectory[k], u)
        #print("状態",next_state)

        # CISmap_data の行数と列数を表示
        #print(f"CISmap_data の要素数: {CISmap_data.shape[0]}")

        #if next_state < min(X_star_CIS) or next_state > max(X_star_CIS):  #安全でない場合
            #x_trajectory[k + 1] = x_trajectory[k]
        #else:
            #x_trajectory[k + 1] = next_state
        x_trajectory[k + 1] = next_state


        if min(X_star_CIS) <= x_trajectory[k + 1] <= max(X_star_CIS):
            CISmap_data = np.append(CISmap_data, 1)   
        else:
            CISmap_data = np.append(CISmap_data, 0) 
    




    return x_trajectory, D, CISmap_data

def plot_results(states, x_trajectories, CIS_results, T, num_steps):
    plt.figure(figsize=(10, 6))
    time_step = 0  # 現在の時間ステップを管理する変数

    for iteration in range(num_steps):
        x_trajectory = x_trajectories[iteration]
        X_star_CIS = CIS_results[iteration]

        # 状態軌道を青色でプロット
        plt.plot(
            range(time_step, time_step + T - 1),
            x_trajectory[:-1],
            color="blue",
            label="State Trajectory" if iteration == 0 else None,
            linewidth=2,
        )

        # 制御不変集合を赤い領域で表示 (右に T ステップシフト)
        if X_star_CIS is not None:
            min_CIS = min(X_star_CIS)
            max_CIS = max(X_star_CIS)

            # シフトした範囲が全体を超えないように調整
            cis_start = time_step + T
            cis_end = min(time_step + 2 * T - 1, num_steps * (T - 1))

            if cis_start < cis_end:  # 表示可能な範囲の場合のみ描画
                plt.fill_between(
                    range(cis_start, cis_end),
                    min_CIS,
                    max_CIS,
                    color="red",
                    alpha=0.3,
                    label="Control Invariant Set" if iteration == 0 else None,
                )

        time_step += T - 1

    plt.title("State Trajectories and Control Invariant Sets", fontsize=14)
    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("State", fontsize=12)
    plt.ylim([states.min(), states.max()])
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()




def main():
    S = 100
    A = 30
    T = 1000
    xmin, xmax = -2, 2
    umin, umax = -1, 3
    states = np.linspace(xmin, xmax, S)
    actions = np.linspace(umin, umax, A)
    d = 300
    num_steps = 3
    x0 = 0
    epsilon = 0.3
    alpha_eta = 1.0
    N = 1

    # Initialize Phi and D
    Phi = np.random.rand(S + 1, A, d)
    mu = np.random.rand(S + 1, d) - 0.5
    invmu = np.linalg.pinv(mu)
    P = transition_probability(S, A, states, actions, xmin, xmax)

    for i in range(S):
        for j in range(A):
            Pcomp = P[i, j, :].reshape(S + 1, 1)
            Phi[i, j, :] = (invmu @ Pcomp).flatten()

    D = np.array([], dtype=np.float32)
    CISmap_data = np.array([], dtype=np.float32)
    x_trajectory_data = np.array([])

    x_trajectories = []  # 状態軌道を保存するリスト
    CIS_results = []  # 制御不変集合を保存するリスト

 


    Omega_initial = np.linspace(xmin, xmax, S)

    for iteration in range(1, num_steps + 1):
        print(f"Iteration {iteration}")

        if iteration == 1:
            # Generate trajectory within the initial safe region (-2, 2)
            x_trajectory, D = generate_initial_trajectory(x0, T, states, actions, Phi, D,d)
            #print("Dの行数:", D.shape[0])  # 行数
            #print("Dの列数:", D.shape[1])  # 列数
            
            print("状態軌道最小値",min(x_trajectory))
            print("状態軌道最大値",max(x_trajectory))
        else:
            # Generate trajectory within the control invariant set
            x_trajectory, D, CISmap_data = generate_trajectory(x0, T, states, actions, Phi, D, X_star_CIS, d, CISmap_data, alpha_eta)
            print(f"Dの行数: {D.shape[0]}, 列数: {D.shape[1]}")
            print("状態軌道最小値",min(x_trajectory))
            print("状態軌道最大値",max(x_trajectory))

        x_trajectories.append(x_trajectory)



           # 状態軌道データを保存
        if x_trajectory_data.size == 0:
            x_trajectory_data = x_trajectory
        else:
            x_trajectory_data = np.hstack([x_trajectory_data, x_trajectory])

        
        print(f"x_trajectory_dataの要素数: {x_trajectory_data.shape[0]}") 
        # Compute the control invariant set
        X_star_CIS, p = compute_CIS(x_trajectory_data, states, actions, xmin, xmax, Omega_initial, N, epsilon, Phi, d, D, alpha_eta,iteration)
        print("p",p)
        print( "確率的制御不変集合",X_star_CIS)

        CIS_results.append(X_star_CIS)
  


        if iteration==1:
            for k in range(len(x_trajectory) - 1):
                if min(X_star_CIS) <= x_trajectory[k + 1] <= max(X_star_CIS):
                    CISmap_data = np.append(CISmap_data, 1)
                else:
                    CISmap_data = np.append(CISmap_data, 0)

    
    # 結果のプロット
    plot_results(states, x_trajectories, CIS_results, T, num_steps)

        
 


if __name__ == "__main__":
    main()
