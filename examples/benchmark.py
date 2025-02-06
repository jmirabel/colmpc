# BSD 3-Clause License
#
# Copyright (C) 2025, LAAS-CNRS.
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.

import os
import time

import create_ocp
import crocoddyl
import mim_solvers
import pinocchio as pin
from param_parsers import ParamParser, argument_parser
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

if args.velocity:
    ocp, objects = create_ocp.create_ocp_velocity(rmodel, cmodel, pp)
else:
    # OCP with distance constraints
    ocp, objects = create_ocp.create_ocp_distance(
        rmodel, cmodel, args.distance_in_cost, pp
    )

ocp.problem.nthreads = args.nthreads

frame_placement_residual = objects["framePlacementResidual"]
ref = frame_placement_residual.reference

if args.verbose:
    ocp.setCallbacks([
        mim_solvers.CallbackLogger(),
        mim_solvers.CallbackVerbose(),
    ])
# ocp.use_filter_line_search = True
ocp.mu_dynamic = -1
ocp.mu_constraint = -1

XS_init = [pp.get_X0()] * (pp.get_T() + 1)
US_init = ocp.problem.quasiStatic(XS_init[:-1])

ocp.xs = XS_init.copy()
ocp.us = US_init.copy()

for n in range(10):
    start = time.time()
    ok = ocp.solve(ocp.xs, ocp.us, 1000)
    stop = time.time()
    print(f"{n:4}  {1 if ok else 0}  {(stop - start)*1e3:6.2f}  {ocp.iter:3}")

    create_ocp.shift_result(ocp, pp.get_dt())

    ref.translation[0] += 0.01
    frame_placement_residual.reference = ref
    if n == 0:
        # ocp.use_filter_line_search = True
        if args.enable_profiler:
            crocoddyl.enable_profiler()

if args.enable_profiler:
    crocoddyl.stop_watch_report(2)