import gym
from gym import spaces
import numpy as np
import math

class CustomEnv(gym.Env):
    def __init__(self,goal_velocity=0):
        super(CustomEnv, self).__init__()
        # 状態空間と行動空間の設定
        self.min_position = -1.5
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity=goal_velocity

        self.force = 0.001
        self.gravity = 0.0025

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)
        n=3#行動の数
        self.action_space = spaces.Discrete(n)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def reset(self):
        self.state = np.array([np.random.uniform(-0.6, -0.4), 0.0], dtype=np.float32)  # 初期位置と速度
        return self.state

    def step(self, action):
        position, velocity = self.state
        n=3#行動の数
        #friction=0.25

        # カスタム状態遷移
        velocity += (action-1)*self.force - self.gravity * math.cos(3*position)
        #velocity += (action-(n-1)/2)*self.force - self.gravity * math.cos(3*position) - friction*velocity
        #velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        #position = np.clip(position, self.min_position, self.max_position)

        # 左端での速度リセット
        #if position == self.min_position and velocity < 0:
        #    velocity = 0

         # 終了条件 位置0.5以上　速度0以上
         #位置-1以下速度0以下
        done = bool(position >= self.goal_position and velocity >=self.goal_velocity)

        #報酬
        reward = -1.0 if not done else 0.0  # 報酬関数

        self.state = np.array([position, velocity], dtype=np.float32)
        #print(f"Step observation: {self.state}, type: {type(self.state)}, shape: {self.state.shape}")
        return self.state, reward, done, {}

    def render(self, mode="human"):
        pass  # 必要に応じて描画を実装
