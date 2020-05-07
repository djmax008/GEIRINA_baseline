"""
__author__ = 'jiajun'
# Copyright (C) 2018 - 2020 GEIRI North America
# Authors: jiajaun <jiajun.duan@geirina.net>
"""

from grid2op.MakeEnv import make
import grid2op
from grid2op.VoltageControler import ControlVoltageFromFile
from grid2op.Rules import AlwaysLegal
from l2rpn_baselines.Geirina import train
from l2rpn_baselines.Geirina import evaluate
from lightsim2grid import LightSimBackend


backend = LightSimBackend()
#env = make2(dataset_path="grid2op/data/rte_case14_redisp", action_class=grid2op.Action.Action, volagecontroler_class=ControlVoltageFromFile, gamerules_class=AlwaysLegal)

env = grid2op.make('l2rpn_case14_sandbox', gamerules_class=AlwaysLegal, voltagecontroler_class=ControlVoltageFromFile)

train(env,
        name="Geirina",
        iterations=10,
        load_path=None,
        save_path="basline_result")
'''
evaluate(env,
        model_name="_model_176_step_04-24-16-39",
        save_path="basline_result",
        logs_path="logs-train",
        nb_episode=2,
        nb_process=1,
        max_steps=6000,
        verbose=False,
        save_gif="basline_result")
'''
