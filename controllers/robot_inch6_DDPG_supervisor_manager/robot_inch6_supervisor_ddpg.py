from deepbots.supervisor import RobotSupervisorEnv
from gym.spaces import Box
import numpy as np
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..')) 

class RobotInch6Supervisor(RobotSupervisorEnv):
    """
    Observation:
        Type: Box(10)
        Num	Observation                Min(rad)      Max(rad)
        0	Target x                   -Inf           Inf
        1	Target y                   -Inf           Inf
        2	Target z                   -Inf           Inf
        3	Position Sensor on Q1      -1.57          1.57
        4	Position Sensor on Q2      -1.57          1.57
        5	Position Sensor on Q3      -1.57          1.57 
    Actions:
        Type: Continuous
        Num	  Min   Max   Desc
        0	  -1    +1    Set the motor position from Q1 to Q1 + (action 0) * 0.05
        1	  -1    +1    Set the motor position from Q2 to Q2 + (action 1) * 0.05
        2	  -1    +1    Set the motor position from Q3 to Q3 + (action 2) * 0.05
    Reward:
        Reward is - 2-norm for every step taken + extra points for getting close enough to the target
    Starting State:
        [Target x, Target y, Target z, 0, 0, 0]
    Episode Termination:
        distance between "END_EFFECTOR" and "TARGET" < 0.02 or reached step limit
        Episode length is greater than 300
        Solved Requirements (average episode score in last 500 episodes > -0.5)
    """

    def __init__(self):
        """
        In the constructor the observation_space and action_space are set and
        references to the various components of the robot are initialized.
        """

        super().__init__()

        # Set up gym spaces
        self.observation_space = Box(low=np.array([-np.inf, -np.inf, -np.inf, -1.57, -1.57, -1.57]),
                                     high=np.array([np.inf,  np.inf,  np.inf,  1.57,  1.57,  1.57]),
                                     dtype=np.float64)
        self.action_space = Box(low=np.array([-1.0, -1.0, -1.0]),
                                high=np.array([1.0,  1.0,  1.0]),
                                dtype=np.float64)

        # Set up various robot components
        self.robot = self.getSelf()

        self.position_sensors = []
        for i in range(3):
            name = 'arm_' + str(i + 1) + '_joint_sensor'
            position_sensor = self.getDevice(name)
            position_sensor.enable(self.timestep)
            self.position_sensors.append(position_sensor)

        self.end_effector = self.getFromDef("END_EFFECTOR")

        self.target = self.getFromDef("TARGET")

        self.motor_list = []
        for i in range(3):
            name = 'arm_' + str(i + 1) + '_joint'
            motor = self.getDevice(name)	 # Get the motor handle #position_sensor1
            motor.setPosition(float('inf'))  # Set starting position
            motor.setVelocity(0.0)  # Zero out starting velocity
            self.motor_list.append(motor)

        self.episode_score = 0
        self.episode_score_list = []
        
        # Set these to ensure that the robot stops moving
        self.motor_position_values = np.zeros(3)
        self.motor_position_values_target = np.zeros(3)
        self.distance = float("inf")
        
        self.cnt_retry = 0

    def get_observations(self):
        """
        This get_observation implementation builds the required observation.
        All values apart are gathered here from the robot and TARGET objects.

        :return: Observation: [Target x, Target y, Target z,
                               Value of Position Sensor on Q1, ..., Value of Position Sensor on Q3]
        :rtype: list
        """
        # process of negotiation
        prec = 0.01
        err = np.absolute(np.array(self.motor_position_values) -
                          np.array(self.motor_position_values_target)) < prec
        if not np.all(err) and self.cnt_retry < 20:
            self.cnt_retry = self.cnt_retry + 1
            return ["WAIT"]
        else:
            self.cnt_retry = 0
       
        target_position = self.target.getPosition()
        message = [i for i in target_position]
        message.extend([i for i in self.motor_position_values])
        return message

    def get_reward(self, action):
        """
        Reward is - 2-norm for every step taken + extra points for getting close enough to the target

        :param action: Not used, defaults to None
        :type action: None, optional
        :return: - 2-norm + extra points
        :rtype: float
        """
        target_position = self.target.getPosition()
        end_effector_position = self.end_effector.getPosition()

        self.distance = np.linalg.norm([target_position[0] - end_effector_position[0],
                                        target_position[1] - end_effector_position[1],
                                        target_position[2] - end_effector_position[2]])
        reward = -self.distance
        
        # Extra points
        if self.distance < 0.01:
            reward = reward + 0.30
        elif self.distance < 0.015:
            reward = reward + 0.20
        return reward

    def is_done(self):
        """
        An episode is done if the distance between "END_EFFECTOR" and "TARGET" < 0.02 
        :return: True if termination conditions are met, False otherwise
        :rtype: bool
        """
        target_position = self.target.getPosition()
        end_effector_position = self.end_effector.getPosition()
        distance = np.linalg.norm([target_position[0] - end_effector_position[0],
                                   target_position[1] - end_effector_position[1],
                                   target_position[2] - end_effector_position[2]])
        
        return distance < 0.02

    def solved(self):
        """
        This method checks whether the Panda goal reaching task is solved, so training terminates.
        Solved condition requires that the average episode score of last 500 episodes is over -0.5.

        :return: True if task is solved, False otherwise
        :rtype: bool
        """
        return (len(self.episode_score_list) > 500 and
               np.mean(self.episode_score_list[-500:]) > -0.5)

    def get_default_observation(self):
        """
        Simple implementation returning the default observation which is a zero vector in the shape
        of the observation space.
        :return: Starting observation zero vector
        :rtype: list
        """
        obs = [0.0 for _ in range(self.observation_space.shape[0])]
        return obs

    def apply_action(self, action):
        """
        This method uses the action list provided, which contains the next action to be executed.
        The message contains 3 float values that are applied on each motor as position.

        :param action: The message the supervisor sent containing the next action.
        :type action: list of float
        """
        # ignore this action and keep moving
        if action[0] == -1 and len(action) == 1:
            for i in range(3):
                self.motor_position_values[i] = self.position_sensors[i].getValue()
                self.motor_list[i].setVelocity(10)
                self.motor_list[i].setPosition(self.motor_position_values_target[i])
            return
        
        ps_value = []
        for i in self.position_sensors:
            ps_value.append(i.getValue())        
        self.motor_position_values = np.array(ps_value)

        for i in range(3):
            motor_position = self.motor_position_values[i] + action[i]
            motor_position = np.clip(motor_position, -1.57, 1.57)
            self.motor_list[i].setVelocity(10)
            self.motor_list[i].setPosition(motor_position)
            self.motor_position_values_target[i] = motor_position

    def get_info(self):
        """
        Dummy implementation of get_info.
        :return: Empty dict
        """
        return {}

    def render(self, mode='human'):
        """
        Dummy implementation of render
        :param mode:
        :return:
        """
        print("render() is not used")
