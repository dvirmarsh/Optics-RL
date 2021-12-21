import numpy as np
import numpy.fft as fft
from gym import Env
from gym.spaces import Box, Discrete
import gym
import torch
import torch.nn as nn
import random
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from stable_baselines3 import TD3
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor


class optic_propagation():
    def __init__(self, distance_laser_SLM,
                 distance_SLM_tishue,
                 distance_tishue_focus_point,
                 wave_length,
                 scale,
                 laser_in_field):
        self.laser_SLM = distance_laser_SLM
        self.SLM_tishue = distance_SLM_tishue
        self.tishue_focus_point = distance_tishue_focus_point
        self.wave_length = wave_length
        self.scale = scale
        self.scale0 = scale
        self.field0 = np.complex128(laser_in_field)
        self.field = self.field0
        #factor = laser_in_field.shape[0]/SLM.shape[0]
        #self.SLM = np.repeat(np.repeat(SLM, factor, axis=1), factor, axis=0)


    "scale operator" \
    "input:a-scale factor"
    def ni(self, a):
        self.scale = self.scale/np.abs(a)
        if a<0:
            self.field = np.flip(np.flip(self.field, 1), 0)
        return self

    "quadratic phase" \
    "input: z - propagation distance"
    def Q(self,z):
        X, Y = np.meshgrid(self.scale, self.scale)
        k = 2*np.pi/self.wave_length
        phase = np.exp(1j*k*(X**2+Y**2)/(2*z))
        self.field = self.field * phase
        return self

    "Fourier operator"
    def F(self):
        self.field = fft.fftshift(fft.fft2(fft.fftshift(self.field)))
        dx = 1 / (self.scale[1] - self.scale[0])
        self.scale = np.linspace(-dx / 2, dx / 2, len(self.scale))
        return self

    "free space propagation operator" \
    "input: z- propagation distance"
    def R_diverges(self,z):
        self.Q(z).F().ni(1/(self.wave_length*z)).Q(z)
        k = 2 * np.pi / self.wave_length
        self.field = self.field*np.exp(1j*k*z)/(1j*self.wave_length*z)
        return self
    "propagation with phase from the SLM and tishue phase" \
    "outputs: the intensity taken by the camera"
    def diverge_and_scatter(self):
        if self.laser_SLM > 0:
            self.R_diverges(self.laser_SLM)
        self.field *= np.exp(1j*self.SLM)
        if self.SLM_tishue > 0:
            self.R_diverges(self.SLM_tishue)
        self.field *= np.exp(1j*self.Tishue_phase)
        if self.tishue_focus_point > 0:
            self.R_diverges(self.tishue_focus_point)
        return np.abs(self.field)**2
    "reset with new SLM and tissue phase"
    def reset(self, SLM, Tissue):
        N = (self.field.shape[0] - SLM.shape[0]) // 2
        self.SLM = np.pad(SLM, N, mode='edge')
        self.field = self.field0
        self.scale = self.scale0
        self.Tishue_phase = Tissue



class optic_env(Env):
    def __init__(self, SLM_size,
                 pic_size,
                 input_length,
                 wave_length,
                 distance_laser_SLM,
                 distance_SLM_tishue,
                 distance_tishue_focus_point,
                 laser_beam_radius,
                 resize=2
                 ):
        self.SLM_size = SLM_size
        self.pic_size = pic_size
        self.wave_length = wave_length
        self.laser_SLM = distance_laser_SLM
        self.SLM_tishue = distance_SLM_tishue
        self.tishue_focus_point = distance_tishue_focus_point
        self.scale = np.linspace(-input_length/2, input_length/2, pic_size[0])
        X, Y = np.meshgrid(self.scale, self.scale)
        self.laser_in_field = np.float32(X**2 + Y**2 <= laser_beam_radius**2)
        N = pic_size[0]
        self.row_idx = [N//2-1, N//2-1, N//2, N//2]
        self.col_idx = [N//2-1, N//2, N//2-1, N//2]
        self.action_space = Box(low=-1, high=1, shape=(np.prod(SLM_size),))
        self.observation_space = Box(low=0, high=255, shape=(*pic_size, 1), dtype=np.uint8)
        self.resize = resize
        self.prop = optic_propagation(self.laser_SLM,
                                      self.SLM_tishue,
                                      self.tishue_focus_point,
                                      self.wave_length,
                                      self.scale,
                                      self.laser_in_field)




    def reset(self):
        self.SLM = np.zeros((self.SLM_size[0]*self.resize, self.SLM_size[0]*self.resize))
        #self.tishue_phase = (np.random.rand(*self.pic_size) -1/2)*2*np.pi

        self.prop.reset(self.SLM, self.tishue_phase)
        obs = self.prop.diverge_and_scatter()
        obs = np.uint8((obs - np.min(obs)) / (np.max(obs) - np.min(obs)) * 255)
        obs = np.expand_dims(obs, axis=-1)
        self.counter = 0
        return obs

    def step(self, delta_phase):
        self.counter += 1
        self.update_tishue_phase()  # update the tissue phase by the given environment
        delta_phase = np.reshape(delta_phase*np.pi, self.SLM_size)  # The actions returned are with the shape of (N^2,)
        # and in the range [-1,1]
        if self.resize > 1:
            delta_phase = cv2.resize(delta_phase, (self.SLM_size[0]*self.resize, self.SLM_size[1]*self.resize),
                                     interpolation=cv2.INTER_CUBIC)
        self.SLM += delta_phase  # update the SLM
        self.prop.reset(self.SLM, self.tishue_phase)
        obs = self.prop.diverge_and_scatter()
        obs = np.uint8((obs-np.min(obs))/(np.max(obs)-np.min(obs))*255)  # normalize the observation
        reward = np.min(obs[self.row_idx, self.col_idx])/np.mean(obs)  # how much is the beam focused
        obs = np.expand_dims(obs, axis=-1)
        done = False
        if self.counter == 100:
            done = True
        info = {}
        return obs, reward, done, info

    def update_tishue_phase(self):
        return

class CircEnv(optic_env):
    def __init__(self, SLM_size,
                 pic_size,
                 input_length,
                 wave_length,
                 distance_laser_SLM,
                 distance_SLM_tishue,
                 distance_tishue_focus_point,
                 laser_beam_radius,
                 d_theta,
                 resize=2
                 ):
        super(CircEnv, self).__init__(SLM_size,
                                      pic_size,
                                      input_length,
                                      wave_length,
                                      distance_laser_SLM,
                                      distance_SLM_tishue,
                                      distance_tishue_focus_point,
                                      laser_beam_radius,
                                      resize)
        self.d_theta = d_theta
        self.N = self.pic_size[0]

    def update_tishue_phase(self):
        Tishue = ndimage.rotate(self.Big_Tishue, self.d_theta*self.counter, reshape=False, order=0)
        self.tishue_phase = Tishue[self.N//2:-self.N//2, self.N//2:-self.N//2]

    def reset(self):
        #self.SLM = np.zeros((self.SLM_size[0]*self.resize, self.SLM_size[0]*self.resize))
        self.Big_Tishue = (np.random.rand(self.N*2, self.N*2) - 1/2)*2*np.pi
        self.tishue_phase = self.Big_Tishue[self.N//2:-self.N//2, self.N//2:-self.N//2]
        obs = super().reset()
        return obs


class BufferWrapper(gym.ObservationWrapper):
    """
    Only every k-th frame is collected by the buffer
    """

    def __init__(self, env, n_steps=3):
        super(BufferWrapper, self).__init__(env)
        old_space = env.observation_space
        self.dtype = old_space.dtype
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=-1),
                                                old_space.high.repeat(n_steps, axis=-1), dtype=old_space.dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:,:,:-1] = self.buffer[:,:,1:]
        self.buffer[:,:,-1] = np.squeeze(observation)
        return self.buffer






config = {
    "policy_type": "CnnPolicy",
    "SLM_size": (10, 10),
    "pic_size": (64, 64),
    "input_length": 2e-2,
    "wave_length": 532e-9,
    "distannce_laser_SLM": 0,
    "distance_SLM_tishue": 4e-2,
    "distance_tishue_focus_point": 4e-2,
    "laser_beam_radius": 1e-3,
    "d_theta": 1,
    "gamma": 0.99,
    "total_timesteps": 25000,
    "env_name": "CartPole-v1",
}
run = wandb.init(project="opticRL",
                 config=config,
                 sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                 )


env = CircEnv(SLM_size=config["SLM_size"],
              pic_size=config["pic_size"],
              input_length=config["input_length"],
              wave_length=config["wave_length"],
              distance_laser_SLM=config["distannce_laser_SLM"],
              distance_SLM_tishue=config["distance_SLM_tishue"],
              distance_tishue_focus_point=config["distance_tishue_focus_point"],
              laser_beam_radius=config["laser_beam_radius"],
              d_theta=config["d_theta"])
env = Monitor(env)
env = BufferWrapper(env)
model = TD3("CnnPolicy", env, buffer_size=int(2e4), gamma=config["gamma"])

"""done = False
obs = env.reset()
obs0 = obs
rewards = []
action = np.zeros((16,))
while done!=True:
    #action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    rewards.append(reward)

imgplot = plt.imshow(obs)
plt.show()"""
eval_env = CircEnv(SLM_size=config["SLM_size"],
                   pic_size=config["pic_size"],
                   input_length=config["input_length"],
                   wave_length=config["wave_length"],
                   distance_laser_SLM=config["distannce_laser_SLM"],
                   distance_SLM_tishue=config["distance_SLM_tishue"],
                   distance_tishue_focus_point=config["distance_tishue_focus_point"],
                   laser_beam_radius=config["laser_beam_radius"],
                   d_theta=config["d_theta"])
eval_env = Monitor(eval_env)
eval_env = BufferWrapper(eval_env)
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

model.learn(total_timesteps=int(1e6),
            callback=WandbCallback(gradient_save_freq=2,
                                   model_save_path=f"models/{run.id}",
                                   )
            )

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

"""done = False
obs = env.reset()
obs0 = obs
rewards = []
while done!=True:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    rewards.append(reward)

imgplot = plt.imshow(obs)
plt.show()
a = 0"""




"""wave_length = 633e-9
N = 1024
scale = np.linspace(-0.005,0.005,N)

[X,Y] = np.meshgrid(scale,scale)
laser_in_field = np.float32(((X-0.6/N)**2+(Y-0.6/N)**2<=1/N**2)&((X-0.6/N)**2+(Y-0.6/N)**2>=0.25/N**2))
SLM = np.zeros((N,N))
Tishue_phase = np.zeros((N,N))
distance_laser_SLM = 0
distance_SLM_tishue = 0.5
distance_tishue_focus_point = 0
circ = optic_propagation(SLM, Tishue_phase, distance_laser_SLM, distance_SLM_tishue, distance_tishue_focus_point, wave_length, scale, laser_in_field)
neg_circ = optic_propagation(SLM, Tishue_phase, distance_laser_SLM, distance_SLM_tishue, distance_tishue_focus_point, wave_length, scale, 1-laser_in_field)
im = circ.diverge_and_scatter()
im = circ.diverge_and_scatter()
imgplot = plt.imshow(im)
plt.show()"""

a=0





