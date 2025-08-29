import numpy as np
from linear_rl.fourier import FourierBasis

class TrueOnlineSarsaLambda:
    # Reference: True Online Temporal-Difference Learning (https://arxiv.org/pdf/1512.04087.pdf)

    def __init__(self, state_space, action_space, basis='fourier', min_max_norm=False, alpha=0.0001, lamb=0.9, gamma=0.99, epsilon=0.05, fourier_order=7, max_non_zero_fourier=2):
        # 学習率 (step size)
        self.alpha = alpha  
        self.lr = alpha  # 学習率 (エイリアスとして)
        # λ (eligibility trace decay factor)
        self.lamb = lamb  
        # 割引率 (discount factor)
        self.gamma = gamma  
        # ε (epsilon-greedy exploration rate)
        self.epsilon = epsilon  

        # 状態空間
        self.state_space = state_space  
        # 状態次元 (例: 2次元状態空間など)
        self.state_dim = self.state_space.shape[0]  
        # 行動空間
        self.action_space = action_space  
        # 行動数
        self.action_dim = self.action_space.n  
        # 状態の正規化の有無
        self.min_max_norm = min_max_norm  

        # 基底関数 (Fourier basis を使用)
        if basis == 'fourier':
            # Fourier 基底関数の初期化
            self.basis = FourierBasis(self.state_space, self.action_space, fourier_order, max_non_zero=max_non_zero_fourier)
            # 基底関数に基づいた学習率
            self.lr = self.basis.get_learning_rates(self.alpha)

        # 基底関数の数
        self.num_basis = self.basis.get_num_basis()

        # Eligibility traces (エピソード内の履歴に基づいた学習を加速)
        self.et = {a: np.zeros(self.num_basis) for a in range(self.action_dim)}

        # 重みベクトル (各行動に対応するQ値の重み)
        self.theta = {a: np.zeros(self.num_basis) for a in range(self.action_dim)}
        #self.theta={0:[0.0,0.0,...,0.0],1:[0.0,0.0,...,0.0],2:[0.0,0.0,...,0.0]}


        # 前回のQ値 (True Online Sarsaで使用)
        self.q_old = None
        # 現在の選択された行動
        self.action = None

    def learn(self, state, action,safeact, reward, next_state, done):
        """
        True Online Sarsa(λ) アルゴリズムによる学習メソッド
        """
        # 現在の状態に対する基底関数の値
        phi = self.get_features(state)
        # 次の状態に対する基底関数の値
        next_phi = self.get_features(next_state)
        # 現在のQ値
        q = self.get_q_value(phi, action)
        # 次の状態のQ値 (エピソード終了時は0)
        if not done:
            next_q = self.get_q_value(next_phi, self.get_action(next_phi,safeact))
        else:
            next_q = 0.0
        # TD誤差
        td_error = reward + self.gamma * next_q - q
        # 初回時の q_old 初期化
        if self.q_old is None:
            self.q_old = q

        # Eligibility traces と 重みベクトルの更新
        for a in range(self.action_dim):
            if a == action:
                # Eligibility trace の更新
                self.et[a] = self.lamb*self.gamma*self.et[a] + phi - (self.lr*self.gamma*self.lamb*np.dot(self.et[a], phi))*phi
                # 重みベクトルの更新
                self.theta[a] += self.lr*(td_error + q - self.q_old)*self.et[a] - self.lr*(q - self.q_old)*phi
            else:
                # 他の行動の trace の減衰
                self.et[a] = self.lamb*self.gamma*self.et[a]
                # 他の行動の重みベクトルの更新
                self.theta[a] += self.lr*(td_error + q - self.q_old)*self.et[a]
        
        # q_old を次のQ値に更新
        self.q_old = next_q
        # エピソード終了時に traces をリセット
        if done:
            self.reset_traces()

    def get_q_value(self, features, action):

        return np.dot(self.theta[action], features)

    def get_features(self, state): 
        if self.min_max_norm:
            # 状態を正規化
            state = (state - self.state_space.low) / (self.state_space.high - self.state_space.low)
        
        # 基底関数の値を返す
        return self.basis.get_features(state)

    def reset_traces(self):
        """
        Eligibility traces をリセットするメソッド
        """
        self.q_old = None
        for a in range(self.action_dim):
            self.et[a].fill(0.0)


        

    def act(self, obs,safeact):
        features = self.get_features(obs)  # 基底関数の値
        return self.get_action(features,safeact)  # 選択された行動を返す
        

    def get_action(self, features,safeact):

        if np.random.rand() < self.epsilon:
            return np.random.choice(safeact)  # ランダムな行動
        else:
            # 各行動のQ値を計算し、最大の行動を選択
            q_values = [self.get_q_value(features, safeact[a]) for a in range(len(safeact))]
            return q_values.index(max(q_values))
            

