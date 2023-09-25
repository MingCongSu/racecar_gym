import gymnasium
import racecar_gym.envs.gym_api

import yaml
from play_and_evaluate import play

scenario = './scenarios/custom.yml'
output_path = './videos/racecar_test_env.mp4'
frame_size = (640,480)

# For custom scenarios:
env = gymnasium.make(
    id='SingleAgentRaceEnv-v0', 
    scenario=scenario,
    render_mode='rgb_array_follow', # optional: 'rgb_array_birds_eye'
    # render_options=dict(width=320, height=240) # optional
)
with open(f'{scenario}','r') as stream:
    config = yaml.load(stream, Loader=yaml.BaseLoader)
print('[Scenario]:\n', yaml.dump(config), end='')
print('[Observation space]:\n',env.observation_space)
print('[Action space]:\n',env.action_space)

# play and record
play(env=env, output_path=output_path)
# close environment
env.close()