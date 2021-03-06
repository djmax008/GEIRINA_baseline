#!/usr/bin/env python3

"""
In this file, explain how to evaluate your agent.
"""

import os
from grid2op.Runner import Runner

from l2rpn_baselines.Geirina.Geirina import Geirina
from l2rpn_baselines.utils.save_log_gif import save_log_gif

def evaluate(env,
             model_name=".",
             save_path = None,
             logs_path=None,
             nb_episode=1,
             nb_process=1,
             max_steps=-1,
             verbose=False,
             save_gif=False,
             **kwargs):
    """
    In order to submit a valid basline, it is mandatory to provide a "evaluate" function with the same signature as this one.

    Parameters
    ----------
    env: :class:`grid2op.Environment.Environment`
        The environment on which the baseline will be evaluated.

    load_path: ``str``
        The path where the model is stored. This is used by the agent when calling "agent.load)

    logs_path: ``str``
        The path where the agents results will be stored.

    nb_episode: ``int``
        Number of episodes to run for the assessment of the performance.
        By default it's 1.

    nb_process: ``int``
        Number of process to be used for the assessment of the performance.
        Should be an integer greater than 1. By defaults it's 1.

    max_steps: ``int``
        Maximum number of timestep each episode can last. It should be a positive integer or -1.
        -1 means that the entire episode is run (until the chronics is out of data or until a game over).
        By default it's -1.

    verbose: ``bool``
        verbosity of the output

    save_gif: ``bool``
        Whether or not to save a gif into each episode folder corresponding to the representation of the said episode.

    kwargs:
        Other key words arguments that you are free to use for either building the agent save it etc.

    Returns
    -------
    ``None``
    """

    TEST_DEV_ENVS = ["l2rpn_case14_sandbox", "rte_case14_realistic", "rte_case14_redisp", 
    "rte_case14_test", "case14_realistic", "case14_redisp","case14_test", "l2rpn_2019"] 
    if env.name not in TEST_DEV_ENVS:
        raise RuntimeError("This baseline only works for: {} ".format(TEST_DEV_ENVS)) 

    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose

    # Create the agent (this piece of code can change)
    agent = Geirina(env.action_space,
                        env.observation_space,
                        name="Gerina",
                        save_path = save_path)

    # Load weights from file (for example)
    agent.load(model_name)

    # Build runner
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=agent
                    )

    # you can do stuff with your model here

    # start the runner
    res = runner.run(path_save=save_path,
                     nb_episode=nb_episode,
                     nb_process=nb_process,
                     max_iter=max_steps,
                     pbar=False)

    # Print summary
    print("Evaluation summary:")
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        msg_tmp = "\tFor chronics located at {}\n".format(chron_name)
        msg_tmp += "\t\t - cumulative reward: {:.6f}\n".format(cum_reward)
        msg_tmp += "\t\t - number of time steps completed: {:.0f} / {:.0f}".format(nb_time_step, max_ts)
        print(msg_tmp)

    #if save_gif:
    #    save_log_gif(save_path, res)


if __name__ == "__main__":
    """
    This is a possible implementation of the eval script.
    """
    from grid2op import make
    from grid2op.Rules import AlwaysLegal

    from lightsim2grid.LightSimBackend import LightSimBackend
    backend = LightSimBackend()

    env = make("l2rpn_case14_sandbox"
               # gamerules_class=AlwaysLegal,
               #backend=backend
               )

    evaluate(env,
             model_name="geirina",
             save_path="basline_result",
             logs_path="logs-train",            
             nb_episode=1,
             nb_process=1,
             max_steps=1000,
             verbose=False,
             save_gif="basline_result")
