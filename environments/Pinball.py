import numpy as np
from pathlib import Path
from glue.environment import BaseEnvironment

from .pinball.env import PinballModel
# import pygame


class Pinball(BaseEnvironment):

    def __init__(self, seed: int, config_file: Path):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.configuration_file = config_file
        # self.render = render

        # if self.render:
        #     pygame.init()
        #     pygame.display.set_caption('Pinball Domain')
        #     self.screen = pygame.display.set_mode([800, 800])

        self.pinball = PinballModel(self.configuration_file, self.rng)


    def start(self):
        self.pinball.reset_ball_to_start_state()
        obs = self.pinball.get_state()

        # if self.render:
        #     self.environment_view = PinballView(self.screen, self.pinball, self.pinball_goals, self.terminal_goal_index)
        
        extra = {'distance_rep': np.array(obs)}

        return obs, extra


    def step(self, action):
        reward = float(self.pinball.take_action(action))
        next_state = np.array(self.pinball.get_state())
        terminated = bool(self.pinball.episode_ended())

        # if self.render:
        #     self.environment_view.blit()
        #     pygame.display.flip()

        extra = {'distance_rep': next_state}

        return reward, next_state, terminated, extra


# class PinballView:
#     """ This class displays a :class:`PinballModel`

#     This class is used in conjunction with the :func:`run_pinballview`
#     function, acting as a *controller*.

#     We use `pygame <http://www.pygame.org/>` to draw the environment.

#     """
#     def __init__(self, screen, model):
#         """
#         :param screen: a pygame surface
#         :type screen: :class:`pygame.Surface`
#         :param model: an instance of a :class:`PinballModel`
#         :type model: :class:`PinballModel`
#         """
#         self.screen = screen
#         self.model = model

#         self.DARK_GRAY = [64, 64, 64]
#         self.LIGHT_GRAY = [232, 232, 232]
#         self.BALL_COLOR = [0, 0, 255]
#         self.TARGET_COLOR = [255, 0, 0]
#         self.GOAL_COLOR = [0, 255, 0]

#         # Draw the background
#         self.background_surface = pygame.Surface(screen.get_size())
#         self.background_surface.fill(self.LIGHT_GRAY)
#         for obs in model.obstacles:
#             pygame.draw.polygon(self.background_surface, self.DARK_GRAY, list(map(self._to_pixels, obs.points)), 0)

#         if self.goals is not None:
#             for g in range(self.goals.num_goals):
#                 goal = self.goals.goals[g]
#                 radius = self.goals.termination_radius
#                 initiation_radius = self.goals.initiation_radius
#                 pygame.draw.circle(
#                     self.background_surface, self.GOAL_COLOR, self._to_pixels(goal), int(radius*self.screen.get_width()))
#                 pygame.draw.circle(
#                     self.background_surface, self.GOAL_COLOR, self._to_pixels(goal), int(initiation_radius*self.screen.get_width()), width=1)

#         if self.goals and self.terminal_goal_index is not None:
#             goal = self.goals.goals[self.terminal_goal_index]
#             radius = self.goals.termination_radius
#             pygame.draw.circle(
#                 self.background_surface, self.TARGET_COLOR, self._to_pixels(goal), int(radius*self.screen.get_width()))
#         else:
#             pygame.draw.circle(
#                 self.background_surface, self.TARGET_COLOR, self._to_pixels(self.model.target_pos), int(self.model.target_rad*self.screen.get_width()))

#     def _to_pixels(self, pt):
#         """ Converts from real units in the 0-1 range to pixel units

#         :param pt: a point in real units
#         :type pt: list
#         :returns: the input point in pixel units
#         :rtype: list

#         """
#         return [int(pt[0] * self.screen.get_width()), int(pt[1] * self.screen.get_height())]

#     def blit(self):
#         """ Blit the ball onto the background surface """
#         self.screen.blit(self.background_surface, (0, 0))
#         pygame.draw.circle(self.screen, self.BALL_COLOR,
#                            self._to_pixels(self.model.ball.position), int(self.model.ball.radius*self.screen.get_width()))


# def run_pinballview(width, height, configuration):
#     """ Controller function for a :class:`PinballView`

#     :param width: The desired screen width in pixels
#     :type widht: int
#     :param height: The desired screen height in pixels
#     :type height: int
#     :param configuration: The path to a configuration file for a :class:`PinballModel`
#     :type configuration: str

#     """
#     # Launch interactive pygame
#     pygame.init()
#     pygame.display.set_caption('Pinball Domain')
#     screen = pygame.display.set_mode([width, height])

#     goals = PinballGoals()
#     environment = PinballModel(configuration)
#     environment_view = PinballView(screen, environment, goals)

#     actions = {pygame.K_d:PinballModel.ACC_X, pygame.K_w:PinballModel.DEC_Y, pygame.K_a:PinballModel.DEC_X, pygame.K_s:PinballModel.ACC_Y}

#     done = False
#     while not done:
#         pygame.time.wait(50)

#         user_action = PinballModel.ACC_NONE

#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 done = True
#             if event.type == pygame.KEYUP or event.type == pygame.KEYDOWN:
#                 user_action = actions.get(event.key, PinballModel.ACC_NONE)

#         if environment.take_action(user_action) == environment.END_EPISODE:
#             done = True

#         # print(environment.ball.position)
#         environment_view.blit()
#         pygame.display.flip()

#     pygame.quit()
