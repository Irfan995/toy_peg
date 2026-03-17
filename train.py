import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################
# ENVIRONMENT WITH OBSTACLE
#############################################

class ContinuousEnv:

    def __init__(self, size=10):

        self.size = size
        self.dt = 0.1
        self.vA = 1.0
        self.vD = 1.2
        self.capture_radius = 0.5

        # obstacle
        self.obs_center = np.array([size/2, size/2])
        self.obs_radius = 2.0

    def reset(self):

        self.attacker = np.array([1.0, np.random.uniform(0, self.size)])
        self.defender = np.array([self.size/2, np.random.uniform(0, self.size)])

        return self.get_state()

    def get_state(self):
        return np.concatenate([self.attacker, self.defender])

    def clip(self, pos):
        return np.clip(pos, 0, self.size)

    def normalize(self, u):
        norm = np.linalg.norm(u)
        return u/norm if norm > 1 else u

    def in_obstacle(self, pos):
        return np.linalg.norm(pos - self.obs_center) <= self.obs_radius

    def project_outside(self, pos):
        direction = pos - self.obs_center
        if np.linalg.norm(direction) == 0:
            direction = np.random.randn(2)

        direction = direction / np.linalg.norm(direction)
        return self.obs_center + direction * (self.obs_radius + 1e-3)

    def safe_move(self, pos, u, speed):

        next_pos = pos + speed * u * self.dt
        next_pos = self.clip(next_pos)

        if self.in_obstacle(next_pos):
            next_pos = self.project_outside(next_pos)

        return next_pos

    def step(self, uA, uD):

        uA = self.normalize(uA)
        uD = self.normalize(uD)

        self.attacker = self.safe_move(self.attacker, uA, self.vA)
        self.defender = self.safe_move(self.defender, uD, self.vD)

        done = False
        reward = 0

        dist_AD = np.linalg.norm(self.attacker - self.defender)
        dist_target = self.size - self.attacker[0]

        # capture
        if dist_AD <= self.capture_radius:
            reward = -1
            done = True

        # attacker reaches goal
        elif self.attacker[0] >= self.size:
            reward = 1
            done = True

        else:
            # reward shaping
            reward += -0.05 * dist_target
            reward += +0.001 * dist_AD

        return self.get_state(), reward, done


#############################################
# NETWORKS
#############################################

class Actor(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,2),
            nn.Tanh()
        )
    def forward(self,x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 4,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
    def forward(self,s,uA,uD):
        return self.net(torch.cat([s,uA,uD],dim=1))


#############################################
# INIT
#############################################

env = ContinuousEnv()
state_dim = 4

actor_A = Actor(state_dim).to(device)
actor_D = Actor(state_dim).to(device)
critic = Critic(state_dim).to(device)

opt_actor_A = optim.Adam(actor_A.parameters(), lr=1e-3)
opt_actor_D = optim.Adam(actor_D.parameters(), lr=1e-3)
opt_critic = optim.Adam(critic.parameters(), lr=1e-3)

gamma = 0.95

#############################################
# TRAINING
#############################################

episodes = 1000
rewards = []
episode_summaries = []

for ep in range(episodes):

    state = env.reset()
    total_reward = 0
    outcome = "max_steps"
    steps_taken = 0

    for t in range(200):

        s = torch.FloatTensor(state).unsqueeze(0).to(device)

        uA = actor_A(s)
        uD = actor_D(s)

        # add exploration noise
        noise = np.random.normal(0, 0.1, size=2)
        uA_np = (uA.detach().cpu().numpy()[0] + noise)
        uD_np = (uD.detach().cpu().numpy()[0] + noise)

        next_state, r, done = env.step(uA_np, uD_np)

        s_next = torch.FloatTensor(next_state).unsqueeze(0).to(device)

        # critic update
        with torch.no_grad():
            uA_next = actor_A(s_next)
            uD_next = actor_D(s_next)
            target = r + gamma * critic(s_next, uA_next, uD_next)

        q_val = critic(s, uA, uD)
        loss_c = (q_val - target).pow(2).mean()

        opt_critic.zero_grad()
        loss_c.backward()
        opt_critic.step()

        # attacker update (maximize)
        loss_a = -critic(s, actor_A(s), actor_D(s)).mean()
        opt_actor_A.zero_grad()
        loss_a.backward(retain_graph=True)
        opt_actor_A.step()

        # defender update (minimize)
        loss_d = critic(s, actor_A(s), actor_D(s)).mean()
        opt_actor_D.zero_grad()
        loss_d.backward()
        opt_actor_D.step()

        state = next_state
        total_reward += r
        steps_taken = t + 1

        if done:
            if r == -1:
                outcome = "captured"
            elif r == 1:
                outcome = "goal_reached"
            break

    rewards.append(total_reward)
    episode_summaries.append(
        {
            "episode": ep + 1,
            "reward": total_reward,
            "steps": steps_taken,
            "attacker": env.attacker.copy(),
            "defender": env.defender.copy(),
            "outcome": outcome,
        }
    )

    if ep % 100 == 0:
        print("Episode:", ep)

def print_episode_block(title, summaries):
    print(f"\n{title}")
    for summary in summaries:
        attacker = np.round(summary["attacker"], 3).tolist()
        defender = np.round(summary["defender"], 3).tolist()
        print(
            f"Episode {summary['episode']:4d} | outcome = {summary['outcome']:12s} "
            f"| reward = {summary['reward']:.4f} | steps = {summary['steps']:3d} "
            f"| attacker = {attacker} | defender = {defender}"
        )


print_episode_block("Final outputs for the first 100 episodes:", episode_summaries[:100])
print_episode_block("Final outputs for the last 100 episodes:", episode_summaries[-100:])


#############################################
# PLOT REWARD
#############################################

smooth_window = 25
smoothed_rewards = np.convolve(
    rewards,
    np.ones(smooth_window) / smooth_window,
    mode='valid'
)

plt.figure()
plt.plot(rewards, alpha=0.3, label="Raw reward")
plt.plot(
    range(smooth_window - 1, len(rewards)),
    smoothed_rewards,
    linewidth=2,
    label=f"Smoothed reward ({smooth_window}-episode moving average)"
)
plt.title("Training Reward (Obstacle)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.show()


#############################################
# VISUALIZATION
#############################################

def rollout_trained_policy(max_steps=100):

    state = env.reset()
    total_reward = 0
    outcome = "max_steps"

    A_path = [env.attacker.copy()]
    D_path = [env.defender.copy()]

    for step in range(max_steps):

        s = torch.FloatTensor(state).unsqueeze(0).to(device)

        uA = actor_A(s).detach().cpu().numpy()[0]
        uD = actor_D(s).detach().cpu().numpy()[0]

        state, r, done = env.step(uA, uD)
        total_reward += r

        A_path.append(env.attacker.copy())
        D_path.append(env.defender.copy())

        if done:
            if r == -1:
                outcome = "captured"
            elif r == 1:
                outcome = "goal_reached"
            break

    return {
        "reward": total_reward,
        "steps": step + 1,
        "outcome": outcome,
        "attacker": env.attacker.copy(),
        "defender": env.defender.copy(),
        "attacker_path": np.array(A_path),
        "defender_path": np.array(D_path),
    }


def print_game_result(result):
    attacker = np.round(result["attacker"], 3).tolist()
    defender = np.round(result["defender"], 3).tolist()
    if result["outcome"] == "goal_reached":
        winner = "Attacker"
    elif result["outcome"] == "captured":
        winner = "Defender"
    else:
        winner = "No winner"
    print("\nTrained policy game result:")
    print(
        f"Winner = {winner} | outcome = {result['outcome']} | reward = {result['reward']:.4f} "
        f"| steps = {result['steps']} | attacker = {attacker} | defender = {defender}"
    )


def visualize(result):

    A_path = result["attacker_path"]
    D_path = result["defender_path"]

    A_path = np.array(A_path)
    D_path = np.array(D_path)

    plt.figure()

    plt.plot(A_path[:,1], A_path[:,0], 'b-o', label="Attacker")
    plt.plot(D_path[:,1], D_path[:,0], 'r-o', label="Defender")

    # obstacle
    circle = plt.Circle(env.obs_center[::-1], env.obs_radius, color='black')
    plt.gca().add_patch(circle)

    plt.xlim(0, env.size)
    plt.ylim(0, env.size)

    plt.legend()
    plt.title(
        f"Trajectory with Obstacle | {result['outcome']} | "
        f"reward = {result['reward']:.3f}"
    )
    plt.show()


game_result = rollout_trained_policy()
print_game_result(game_result)
visualize(game_result)
