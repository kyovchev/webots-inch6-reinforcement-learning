from numpy import convolve, ones, mean, random

from robot_inch6_supervisor_ppo import RobotInch6Supervisor
from agent.PPO_agent import PPOAgent, Transition

from robot_inch6_PPO_supervisor_manager import EPISODE_LIMIT, STEPS_PER_EPISODE

def run():
    env = RobotInch6Supervisor()
    agent = PPOAgent(number_of_inputs=env.observation_space.shape[0],
                     number_of_actor_outputs=env.action_space.n)

    solved = False
    episode_count = 0
    # Run outer loop until the episodes limit is reached or the task is solved
    while not solved and episode_count < EPISODE_LIMIT:
        observation = env.reset()  # Reset robot and get starting observation
        env.episode_score = 0

        # Move the TARGET to a random position
        env.target = env.getFromDef("TARGET")
        translation_field = env.target.getField('translation')
        pos = [random.randint(9, 15, 1)[0] / 100,
               random.randint(-9, 9, 1)[0] / 100,
               0.7]
        translation_field.setSFVec3f(pos)

        for step in range(STEPS_PER_EPISODE):
            # In training mode the agent samples from the probability distribution,
            # naturally implementing exploration
            selected_action, action_prob = agent.work(observation, type_="selectAction")
            # Step the supervisor to get the current selected_action's reward, the new observation
            # and whether we reached the done condition
            new_observation, reward, done, _ = env.step([selected_action])
            # process of negotiation
            while new_observation == ["WAIT"]:
                new_observation, reward, done, _ = env.step([-1])
            # Save the current state transition in agent's memory
            trans = Transition(observation, selected_action, action_prob, reward, new_observation)
            agent.store_transition(trans)

            if done:
                # Save the episode's score
                env.episode_score_list.append(env.episode_score)
                agent.train_step(batch_size=step + 1)
                solved = env.solved()  # Check whether the task is solved
                break

            env.episode_score += reward  # Accumulate episode reward
            observation = new_observation  # observation for next step is current step's new_observation

        print("Episode #", episode_count, "score:", env.episode_score)
        file = open("./exports/Episode-score.txt","a")
        file.write(str(env.episode_score) + '\n')
        file.close()
        episode_count += 1  # Increment episode counter

    if not solved:
        print("Task is not solved, deploying agent for testing...")
    elif solved:
        print("Task is solved, deploying agent for testing...")

    observation = env.reset()
    env.episode_score = 0.0
    while True:
        selected_action, action_prob = agent.work(observation, type_="selectAction")
        observation, _, done, _ = env.step([selected_action])
        # process of negotiation
        while observation == ["WAIT"]:
            observation, _, done, _ = env.step([-1])
        # Save the current state transition in agent's memory
        step = step + 1
        if done or step == STEPS_PER_EPISODE - 1:
            observation = env.reset()
            step = 0
            # Move the TARGET to a random position
            env.target = env.getFromDef("TARGET")
            translation_field = env.target.getField('translation')
            pos = [random.randint(9, 15, 1)[0] / 100,
                   random.randint(-9, 9, 1)[0] / 100,
                   0.7]
            translation_field.setSFVec3f(pos)