#!/usr/bin/env python3

from l2rpn_baselines.Geirina.Geirina import Geirina

def train(env,
          name="Geirina",
          iterations=1,
          save_path=None,
          load_path=None
          ):
    """
    This an example function to train a baseline.

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

    baseline = Geirina(env.action_space,
                        env.observation_space,
                        name=name,
                        save_path = save_path)

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
    from grid2op.MakeEnv import make, make2
    import grid2op
    from grid2op.VoltageControler import ControlVoltageFromFile
    from grid2op.Rules import AlwaysLegal


    env = make2(dataset_path="grid2op/data/rte_case14_redisp", action_class=grid2op.Action.Action, volagecontroler_class=ControlVoltageFromFile, gamerules_class=AlwaysLegal)

    res = train(env,
            name="Geirina",
            iterations=1,
            load_path=None,
            save_path="basline_result")