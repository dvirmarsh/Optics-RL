import numpy as np
import numpy.fft as fft
from gym import Env
from gym.spaces import Box, Discrete
import torch
import torch.nn as nn
import random
from tqdm import tqdm
import cv2

class optic_propagation():
    def __init__(self,SLM,
                 Tishue_phase,
                 distance_laser_SLM,
                 distance_SLM_tishue,
                 distance_tishue_focus_point,
                 wave_length,
                 scale,
                 laser_in_field):
        self.Tishue_phase = Tishue_phase
        self.laser_SLM = distance_laser_SLM
        self.SLM_tishue = distance_SLM_tishue
        self.tishue_focus_point = distance_tishue_focus_point
        self.wave_length = wave_length
        self.scale = scale
        self.field = laser_in_field
        factor = laser_in_field.shape[0]/SLM.shape[0]
        self.SLM = np.repeat(np.repeat(SLM, factor, axis=1), factor, axis=0)

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
        self.R_diverges(self.laser_SLM)
        self.field *= self.SLM
        self.R_diverges(self.SLM_tishue)
        self.field *= self.Tishue_phase
        self.R_diverges(self.tishue_focus_point)
        return np.abs(self.field)**2



class optic_env(Env):
    def __init__(self, SLM_size,
                 pic_size,
                 input_length,
                 wave_length,
                 distance_laser_SLM,
                 distance_SLM_tishue,
                 distance_tishue_focus_point,
                 laser_beam_radius
                 ):
        self.SLM_size = SLM_size
        self.pic_size = pic_size
        self.wave_length = wave_length
        self.laser_SLM = distance_laser_SLM
        self.SLM_tishue = distance_SLM_tishue
        self.tishue_focus_point = distance_tishue_focus_point
        self.scale = np.linspace(-input_length/2, input_length/2, pic_size[0])
        X, Y = np.meshgrid(self.scale, self.scale)
        self.laser_in_field = np.float(X**2 + Y**2 <= laser_beam_radius**2)
        N = pic_size[0]
        self.row_idx = [N/2-1, N/2-1, N/2, N/2]
        self.col_idx = [N/2-1, N/2, N/2-1, N/2]
        self.action_space = Box(low=-np.pi, high=np.pi, shape=SLM_size)
        self.observation_space = Box(low=0, high=1, shape=pic_size)




    def reset(self):
        self.SLM = np.ones(self.SLM_size)
        self.tishue_phase = (np.random.rand(self.pic_size) -1/2)*2*np.pi
        prop = optic_propagation(self.SLM,
                                 self.tishue_phase,
                                 self.laser_SLM,
                                 self.SLM_tishue,
                                 self.tishue_focus_point,
                                 self.wave_length,
                                 self.scale,
                                 self.laser_in_field)
        return prop.diverge_and_scatter()

    def step(self, delta_phase):
        self.update_tishue_phase()
        self.SLM *= np.exp(1j*delta_phase)
        prop = optic_propagation(self.SLM,
                                 self.tishue_phase,
                                 self.laser_SLM,
                                 self.SLM_tishue,
                                 self.tishue_focus_point,
                                 self.wave_length,
                                 self.scale,
                                 self.laser_in_field)
        obs = prop.diverge_and_scatter()
        reward = np.mean(obs[self.row_idx, self.col_idx])


        return obs, reward, 0









