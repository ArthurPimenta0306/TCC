
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np



def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)
class EnvYaw(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }
    # no caso o environment é o drone
    def __init__(self, Jxx=0.01817, Jyy=0.01883, Jzz=0.03572, target=np.pi / 6):
        super(EnvYaw, self).__init__()

        # State
        self.state = np.array([0, 0]) #Yaw, Yawdot

        self.l = 0.28  # distancia do motor ao centro de massa

        # momentos de inercia
        self.Jxx = Jxx
        self.Jyy = Jyy
        self.Jzz = Jzz


        # velocidade maxima do motor
        self.MaxSpeed = 700

        #limites de velocidade angular do quad
        self.MaxAngularVel = 5.24 #rad/s based on reference "Reinforcement Learning for UAV Attitude Control"


        # %Throttle Factor
        self.b = 1.0927 * 10 ** (-5)
        # %Drag Factor
        self.d = 3.7343 * 10 ** (-7)

        # passo
        self.Ts = 0.01

        # target
        self.target = target

        #maximo torque
        #self.max_torque = self.l * self.b * (self.MaxSpeed **2)
        self.max_torque = 2 * self.d * (self.MaxSpeed **2)


        # Action space são os 2 torques que podem ser de -1.5 a 1.5
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )

        high = np.array([1., 1., self.MaxAngularVel], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high,high = high,dtype = np.float32)
        self.seed()
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # reseta o estado
        high = np.array([np.pi, self.MaxAngularVel])
        self.state = self.np_random.uniform(low=-high, high=high)

        # Return initial environment state variables as logged signals.
        return self._get_obs()

    def _get_obs(self):
        #Roll = self.state[0]
        #Rolldot = self.state[1]

        #return np.array([np.cos(Roll), np.sin(Roll), Rolldot])
        Yaw = self.state[0]
        Yawdot = self.state[1]

        return np.array([np.cos(Yaw), np.sin(Yaw), Yawdot])


    def step(self, Action):
        # Entra acao e estado atual,
        # Retorna Proxima observacao,Reward,IsDone (bool que fala se acabou o episodio) e State

        # This function applies the given action to the environment and evaluates
        # the system dynamics for one simulation step.
        # Define the environment constants.

        # Sample time
        Ts = self.Ts
        # distance from an propeller to the center
        l = self.l

        # %Throttle Factor
        b = self.b
        # %Drag Factor
        d = self.d

        # inertia
        Jxx = self.Jxx
        Jyy = self.Jyy
        Jzz = self.Jzz

        My = np.clip(Action, -self.max_torque, self.max_torque)[0]
        Mz = np.clip(Action, -self.max_torque, self.max_torque)[0]


        # Unpack the state vector from the logged signals.
        #Roll = self.state[0]
        #RollDot = self.state[1]
        Yaw = self.state[0]
        YawDot = self.state[1]

        # Dynamic equations
        #RollDotDot = My / Jyy
        YawDotDot = Mz / Jzz

        # Perform Euler integration.
        #NewRollDot = RollDot + Ts * RollDotDot
        #NewRoll = Roll + Ts * NewRollDot
        NewYawDot = YawDot + Ts * YawDotDot
        NewYaw = Yaw + Ts * NewYawDot


        #self.state = np.array([NewRoll,NewRollDot])
        self.state = np.array([NewYaw,NewYawDot])
        #self.state = self.state + Ts * np.array([RollDot, RollDotDot, PitchDot, PitchDotDot, YawDot, YawDotDot])

        # Transform state to observation.
        NextObs = self.state


        # Get reward. Based on https://towardsdatascience.com/how-to-train-your-quadcopter-adventures-in-machine-learning-algorithms-e6ee5033fd61
        target = self.target

        angularError = NewYaw - target
        angularVel = NewYawDot
        torque = Mz
        reward = -(angle_normalize(angularError) ** 2 + .1 * angularVel ** 2 + .001 * (torque ** 2)) #penaliza erro angular, velocidade angular e torque
        info = {}

        #return NextObs.astype(np.float32), Reward, IsDone, info
        #It is never done.
        return self._get_obs(), reward, False, {}


    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        yaw = self.state[0]

        print("Yaw: ", yaw)


    def close(self):
        pass


class EnvRoll(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    # no caso o environment é o drone
    def __init__(self, Jxx=0.01817, Jyy=0.01883, Jzz=0.03572, target=np.pi / 6):
        super(EnvRoll, self).__init__()

        # State
        self.state = np.array([0, 0])  # roll, rolldot

        self.l = 0.28  # distancia do motor ao centro de massa

        # momentos de inercia
        self.Jxx = Jxx
        self.Jyy = Jyy
        self.Jzz = Jzz


        # velocidade maxima do motor
        self.MaxSpeed = 700

        # limites de velocidade angular do quad
        self.MaxAngularVel = 5.24  # rad/s based on reference "Reinforcement Learning for UAV Attitude Control"

        # %Throttle Factor
        self.b = 1.0927 * 10 ** (-5)
        # %Drag Factor
        self.d = 3.7343 * 10 ** (-7)

        # passo
        self.Ts = 0.01

        # target
        self.target = target

        # maximo torque
        # self.max_torque = self.l * self.b * (self.MaxSpeed **2)
        self.max_torque = 2 * self.d * (self.MaxSpeed ** 2)

        # Action space são os 2 torques que podem ser de -1.5 a 1.5
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )

        high = np.array([1., 1., self.MaxAngularVel], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # reseta o estado
        high = np.array([np.pi, self.MaxAngularVel])
        self.state = self.np_random.uniform(low=-high, high=high)

        # Return initial environment state variables as logged signals.
        return self._get_obs()

    def _get_obs(self):
         Roll = self.state[0]
         Rolldot = self.state[1]

         return np.array([np.cos(Roll), np.sin(Roll), Rolldot])


    def step(self, Action):
        # Entra acao e estado atual,
        # Retorna Proxima observacao,Reward,IsDone (bool que fala se acabou o episodio) e State

        # This function applies the given action to the environment and evaluates
        # the system dynamics for one simulation step.
        # Define the environment constants.

        # Sample time
        Ts = self.Ts

        # inertia
        Jxx = self.Jxx
        Jyy = self.Jyy
        Jzz = self.Jzz

        My = np.clip(Action, -self.max_torque, self.max_torque)[0]

        # Unpack the state vector from the logged signals.
        Roll = self.state[0]
        RollDot = self.state[1]


        # Dynamic equations
        RollDotDot = My / Jyy

        # Perform Euler integration.
        NewRollDot = RollDot + Ts * RollDotDot
        NewRoll = Roll + Ts * NewRollDot

        self.state = np.array([NewRoll,NewRollDot])

        # Transform state to observation.
        NextObs = self.state

        # Get reward. Based on https://towardsdatascience.com/how-to-train-your-quadcopter-adventures-in-machine-learning-algorithms-e6ee5033fd61
        target = self.target

        angularError = NewRoll - target
        angularVel = NewRollDot
        torque = My
        reward = -(angle_normalize(angularError) ** 2 + .1 * angularVel ** 2 + .001 * (
                    torque ** 2))  # penaliza erro angular, velocidade angular e torque
        info = {}

        # It is never done.
        return self._get_obs(), reward, False, {}

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        roll = self.state[0]

        print("Roll: ", roll)

    def close(self):
        pass


class EnvPitch(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    # no caso o environment é o drone
    def __init__(self, Jxx=0.01817, Jyy=0.01883, Jzz=0.03572, target=np.pi / 6):
        super(EnvPitch, self).__init__()

        # State
        self.state = np.array([0, 0])  # roll, rolldot

        self.l = 0.28  # distancia do motor ao centro de massa

        # momentos de inercia
        self.Jxx = Jxx
        self.Jyy = Jyy
        self.Jzz = Jzz

        # velocidade maxima do motor
        self.MaxSpeed = 700

        # limites de velocidade angular do quad
        self.MaxAngularVel = 5.24  # rad/s based on reference "Reinforcement Learning for UAV Attitude Control"

        # %Throttle Factor
        self.b = 1.0927 * 10 ** (-5)
        # %Drag Factor
        self.d = 3.7343 * 10 ** (-7)

        # passo
        self.Ts = 0.01

        # target
        self.target = target

        # maximo torque
        # self.max_torque = self.l * self.b * (self.MaxSpeed **2)
        self.max_torque = 2 * self.d * (self.MaxSpeed ** 2)

        # Action space são os 2 torques que podem ser de -1.5 a 1.5
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )

        high = np.array([1., 1., self.MaxAngularVel], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # reseta o estado
        high = np.array([np.pi, self.MaxAngularVel])
        self.state = self.np_random.uniform(low=-high, high=high)

        # Return initial environment state variables as logged signals.
        return self._get_obs()

    def _get_obs(self):
        Pitch = self.state[0]
        Pitchdot = self.state[1]

        return np.array([np.cos(Pitch), np.sin(Pitch), Pitchdot])

    def step(self, Action):
        # Entra acao e estado atual,
        # Retorna Proxima observacao,Reward,IsDone (bool que fala se acabou o episodio) e State

        # This function applies the given action to the environment and evaluates
        # the system dynamics for one simulation step.
        # Define the environment constants.

        # Sample time
        Ts = self.Ts

        # inertia
        Jxx = self.Jxx
        Jyy = self.Jyy
        Jzz = self.Jzz

        Mx = np.clip(Action, -self.max_torque, self.max_torque)[0]

        # Unpack the state vector from the logged signals.
        Pitch = self.state[0]
        PitchDot = self.state[1]

        # Dynamic equations
        PitchDotDot = Mx / Jxx

        # Perform Euler integration.
        NewPitchDot = PitchDot + Ts * PitchDotDot
        NewPitch = Pitch + Ts * NewPitchDot

        self.state = np.array([NewPitch, NewPitchDot])

        # Transform state to observation.
        NextObs = self.state

        # Get reward. Based on https://towardsdatascience.com/how-to-train-your-quadcopter-adventures-in-machine-learning-algorithms-e6ee5033fd61
        target = self.target

        angularError = NewPitch - target
        angularVel = NewPitchDot
        torque = Mx
        reward = -(angle_normalize(angularError) ** 2 + .1 * angularVel ** 2 + .001 * (
                torque ** 2))  # penaliza erro angular, velocidade angular e torque
        info = {}

        # It is never done.
        return self._get_obs(), reward, False, {}

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot
        roll = self.state[0]

        print("Roll: ", roll)

    def close(self):
        pass


