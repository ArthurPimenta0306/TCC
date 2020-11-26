#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 15:38:09 2020

@author: arthurmoriggipimenta
"""

import numpy as np
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2, common
import matplotlib.pyplot as plt
from stable_baselines.common.callbacks import BaseCallback
import os
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
from stable_baselines import results_plotter


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


def main():
    #criando diretorio
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    #criando envs
    envRoll = gym.make('gym_foo:DroneRoll-v0')
    envRoll = Monitor(envRoll, log_dir)
    modelRoll = PPO2(MlpPolicy, envRoll, gamma=0.99, n_steps=2048, ent_coef=0.0,
                 learning_rate=3e-4, lam=0.95,
                 nminibatches=32, noptepochs=10, cliprange=0.2,
                 verbose=1)
    envPitch = gym.make('gym_foo:DronePitch-v0')
    envPitch = Monitor(envPitch, log_dir)

    modelPitch = PPO2(MlpPolicy, envPitch, gamma=0.99, n_steps=2048, ent_coef=0.0,
                     learning_rate=3e-4, lam=0.95,
                     nminibatches=32, noptepochs=10, cliprange=0.2,
                     verbose=1)
    envYaw = gym.make('gym_foo:DroneYaw-v0')
    envYaw = Monitor(envYaw, log_dir)

    modelYaw = PPO2(MlpPolicy, envYaw, gamma=0.99, n_steps=2048, ent_coef=0.0,
                     learning_rate=3e-4, lam=0.95,
                     nminibatches=32, noptepochs=10, cliprange=0.2,
                     verbose=1)
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    #treinando
    time_steps = 2e6
    modelRoll.learn(total_timesteps=int(2e6), callback=callback)
    results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "PPO Roll")
    plt.show()

    modelPitch.learn(total_timesteps=int(2e6), callback=callback)
    results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "PPO Pitch")
    plt.show()

    modelYaw.learn(total_timesteps=int(2e6), callback=callback)
    results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "PPO Yaw")
    plt.show()

    #salvando modelos
    modelRoll.save("Drone_Roll_PPO_001")
    modelPitch.save("Drone_Pitch_PPO_001")
    modelYaw.save("Drone_Yaw_PPO_001")

    #Load modelo
    #model = PPO2.load("Drone_Roll_PPO_0.01")


    #testando gerando resposta no tempo
    T = [0]
    # Loop de teste
    t = 0
    #obs = env.reset()
    obsRoll = envRoll.reset()
    obsPitch = envPitch.reset()
    obsYaw = envYaw.reset()
    Roll = [envRoll.state[0]]
    Pitch = [envPitch.state[0]]
    Yaw = [envYaw.state[0]]

    #loop de simulação
    while t < 10:  # ate 10 segundos

        actionRoll, _states = modelRoll.predict(obsRoll)
        # Retrieve new state, reward, and whether the state is terminal
        obsRoll, reward, done, info = envRoll.step(actionRoll)
        Roll.append((180/np.pi)*envRoll.state[0])

        actionPitch, _states = modelPitch.predict(obsPitch)
        # Retrieve new state, reward, and whether the state is terminal
        obsPitch, reward, done, info = envPitch.step(actionPitch)
        Pitch.append((180 / np.pi) * envPitch.state[0])

        actionYaw, _states = modelYaw.predict(obsYaw)
        # Retrieve new state, reward, and whether the state is terminal
        obsYaw, reward, done, info = envYaw.step(actionYaw)
        Yaw.append((180 / np.pi) * envYaw.state[0])

        t += 0.01
        T.append(t)


    #Plots
    plt.figure(1)
    plt.plot(T, Roll)
    plt.yticks(np.arange(0, 190, 10))
    plt.ylabel('Roll')
    plt.xlabel('Time (seconds)')
    plt.title('Roll Response')
    plt.grid()
    plt.show()

    plt.figure(2)
    plt.plot(T, Pitch)
    plt.yticks(np.arange(0, 190, 10))
    plt.ylabel('Pitch')
    plt.xlabel('Time (seconds)')
    plt.title('Pitch Response')
    plt.grid()
    plt.show()

    plt.figure(3)
    plt.plot(T, Yaw)
    plt.yticks(np.arange(0, 190, 10))
    plt.ylabel('Yaw')
    plt.xlabel('Time (seconds)')
    plt.title('Yaw Response')
    plt.grid()
    plt.show()

main()