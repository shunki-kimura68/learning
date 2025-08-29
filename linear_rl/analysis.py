import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 山の形状を定義
x = np.linspace(-1.2, 0.6, 400)
y = np.sin(3 * x) * 0.4  # 山の傾斜

# ゴールの位置
goal_x = 0.5  # 右側の山の頂上
goal_y = np.sin(3 * goal_x) * 0.4 + 0.05  # 山の高さに合わせる

# 初期パラメータ（学習前）
cart_x_before = -0.6  # 初期位置（谷の底）
velocity_before = 0.01  # 小さな振り子運動

# 初期パラメータ（学習後）
cart_x_after = -0.6  # 初期位置（谷の底）
velocity_after = 0.058  # 大きく振りながら登る

gravity = 0.0025  # 重力（エネルギー減衰なし）

amplitude_after=0.2

# === 学習前のプロット ===
fig1, ax1 = plt.subplots(figsize=(6, 5))
ax1.plot(x, y, 'k')  # 山の曲線
cart_before, = ax1.plot(cart_x_before, np.sin(3 * cart_x_before) * 0.4 + 0.05, 'rs', markersize=10, label="Car (Before Learning)")
ax1.scatter(goal_x, goal_y, marker='^', color='gold', s=200, label="Goal")  # 旗アイコンを追加
ax1.set_xlim(-1.2, 0.6)
ax1.set_ylim(-0.5, 0.5)
ax1.set_xlabel("Position")
ax1.set_ylabel("Height")
ax1.legend(loc="upper left")
ax1.set_title("Before Learning")

# === 学習後のプロット ===
fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(x, y, 'k')  # 山の曲線
cart_after, = ax2.plot(cart_x_after, np.sin(3 * cart_x_after) * 0.4 + 0.05, 'bs', markersize=10, label="Car (After Learning)")
ax2.scatter(goal_x, goal_y , marker='^', color='gold', s=200, label="Goal") # 旗アイコンを追加
ax2.set_xlim(-1.2, 0.6)
ax2.set_ylim(-0.5, 0.5)
ax2.set_xlabel("Position")
ax2.set_ylabel("Height")
ax2.legend(loc="upper left")
ax2.set_title("After Learning")

# 更新関数（学習前）
def update_before(frame):
    global cart_x_before, velocity_before
    #velocity_before += -gravity * np.cos(3 * cart_x_before)  # 山の傾きに応じて加速
    cart_x_before += velocity_before  # 位置を更新

    # 振り子のように振動（範囲を制限）
    if cart_x_before > -0.4 or cart_x_before < -0.6:
        velocity_before *= -1  # 速度を反転

    # カートの高さを更新
    cart_y_before = np.sin(3 * cart_x_before) * 0.4 + 0.05
    cart_before.set_data(cart_x_before, cart_y_before)
    return cart_before,

# 更新関数（学習後）
def update_after(frame):
    global cart_x_after, velocity_after, amplitude_after

    # 振り子運動を維持しながら徐々にゴール方向へ振幅を拡大
    cart_x_after += velocity_after  # 一定速度で移動
    if cart_x_after >= -0.2 + amplitude_after or cart_x_after <= -0.6:
        velocity_after *= -1  # 速度を反転

    # 振幅を広げていく（ゴール方向へ近づける）
    if amplitude_after < 0.6:
        amplitude_after += 0.01

    # カートの高さを更新
    cart_y_after = np.sin(3 * cart_x_after) * 0.4 + 0.05
    cart_after.set_data(cart_x_after, cart_y_after)
    return cart_after,

# アニメーション設定
ani_before = animation.FuncAnimation(fig1, update_before, frames=200, interval=50, blit=True)
ani_after = animation.FuncAnimation(fig2, update_after, frames=200, interval=50, blit=True)

# 表示
plt.show()
