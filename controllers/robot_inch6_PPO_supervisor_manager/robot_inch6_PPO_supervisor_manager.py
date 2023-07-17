"""
More runners for RL algorithms can be added here.
"""
import os
import PPO_runner

# Modify these constants if needed.
STEPS_PER_EPISODE = 150
EPISODE_LIMIT = 15000

def create_path(path):
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

if __name__ == '__main__':
    # dest of traing trend (text file & trend plot)
    create_path("./exports/")

    # pass a path to load the pretrained models, and pass "" for training from scratch
    PPO_runner.run()