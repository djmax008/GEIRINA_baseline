#!/usr/bin/env python3

from l2rpn_baselines.Geirina.Geirina import Geirina

def train(env,
          name="Geirina",
          iterations=1000000,
          save_path=None,
          load_path=None
          ):
    """
    This an example function to train a baseline.

    NB： This baaseline only works for IEEE14 bus environment. For other environments, the agent should be modified accordingly.

    In order to be valid, if you chose (which is recommended) to provide a training script to help other retrain your
    baseline in different environments, or for longer period of time etc. This script should be contain the "train"
    function with at least the following arguments.

    Parameters
    ----------
    env: :class:`grid2op.Environment.Environment`
        The environmnent on which the baseline will be trained

    name: ``str``
        Fancy name you give to this baseline.

    iterations: ``int``
        Number of training iterations to perform

    save_path: ``str``
        The path where the baseline will be saved at the end of the training procedure.

    load_path: ``str``
        Path where to look for reloading the model. Use ``None`` if no model should be loaded.

    kwargs:
        Other key-word arguments that you might use for training.

    """
    TEST_DEV_ENVS = ["l2rpn_case14_sandbox", "rte_case14_realistic", "rte_case14_redisp", 
    "rte_case14_test", "case14_realistic", "case14_redisp","case14_test", "l2rpn_2019"] 
    if env.name not in TEST_DEV_ENVS:
        raise RuntimeError("This baseline only works for: {} ".format(TEST_DEV_ENVS)) 

    baseline = Geirina(env.action_space,
                        env.observation_space,
                        name=name,
                        save_path=save_path)

    if load_path is not None:
        baseline.load(load_path)

    baseline.train(env, iterations)
    # as in our example (and in our explanation) we recommend to save the mode regurlarly in the "train" function
    # it is not necessary to save it again here. But if you chose not to follow these advice, it is more than
    # recommended to save the "baseline" at the end of this function with:
    # baseline.save(path_save)


if __name__ == "__main__":
    """
    This is a possible implementation of the train script.
    """
    from grid2op import make
    from grid2op.Rules import AlwaysLegal

    from lightsim2grid.LightSimBackend import LightSimBackend
    backend = LightSimBackend()

    env = make("l2rpn_case14_sandbox"
               # gamerules_class=AlwaysLegal,
               #backend=backend
               )

    train(env,
          name="Geirina",
          iterations=1000000, # this is apparently not used !
          save_path="basline_result")
