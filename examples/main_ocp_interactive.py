# BSD 3-Clause License
#
# Copyright (C) 2025, LAAS-CNRS.
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.

import argparse
import os

import create_ocp
import pinocchio as pin
import simulation
import mim_solvers
from param_parsers import ParamParser, argument_parser
from visualizer import create_viewer
from wrapper_panda import PandaWrapper

# Parse the arguments
parser = argument_parser()
args = parser.parse_args()


# Creating the robot
robot_wrapper = PandaWrapper(capsule=False)
rmodel, cmodel, vmodel = robot_wrapper()

yaml_path = os.path.join(os.path.dirname(__file__), "scenes.yaml")
pp = ParamParser(yaml_path, args.scene)

cmodel = pp.add_collisions(rmodel, cmodel)

cdata = cmodel.createData()
rdata = rmodel.createData()

# Generating the meshcat visualizer
goal_frame_id = rmodel.addFrame(
    pin.Frame("goal", 0, 0, pp.get_target_pose(), pin.FrameType.OP_FRAME)
)
vis = create_viewer(rmodel, cmodel, cmodel)
vis.displayFrames(True, [rmodel.getFrameId("panda2_hand_tcp"), goal_frame_id])
# add_sphere_to_viewer(
# vis, "goal", 5e-2, pp.get_target_pose().translation, color=0x006400
# )

if args.velocity:
    ocp, objects = create_ocp.create_ocp_velocity(rmodel, cmodel, pp)
else:
    # OCP with distance constraints
    ocp, objects = create_ocp.create_ocp_distance(
        rmodel, cmodel, args.distance_in_cost, pp
    )
ocp.problem.nthreads = args.nthreads
if args.verbose:
    ocp.setCallbacks([
        mim_solvers.CallbackLogger(),
        mim_solvers.CallbackVerbose(),
    ])

# ocp.use_filter_line_search = True
ocp.mu_dynamic = -1
ocp.mu_constraint = -1

simulation.simulation_loop(
    ocp,
    rmodel,
    goal_frame_id,
    cmodel,
    cmodel.ngeoms - 1,
    objects["framePlacementResidual"],
    pp,
    vis,
)
