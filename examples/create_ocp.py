# BSD 3-Clause License
#
# Copyright (C) 2024, LAAS-CNRS.
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.

import crocoddyl
import mim_solvers
import numpy as np
import pinocchio as pin
from param_parsers import ParamParser

import colmpc as col

def shift_result(ocp, dt):
    p = ocp.problem
    for i, (m, d) in enumerate(zip(p.runningModels, p.runningDatas)):
        if m.dt == dt:
            ocp.xs[i] = ocp.xs[i+1]
            # for the last running model, i+1 is the terminal model.
            # There is no control for this one. The result of the current loop is
            # that if two last control will be equal.
            if i < len(p.runningModels) - 1:
                ocp.us[i] = ocp.us[i+1]
        else:
            assert m.dt > dt
            mdt = m.dt
            m.dt = dt
            m.calc(d, ocp.xs[i], ocp.us[i])
            m.dt = mdt
            ocp.xs[i] = d.xnext
            # Keep the same control because we are still in the segment where
            # ocp.us[i] was to be applied.
            # TODO any better guess ? e.g.
            # - weighted average of ocp.us[i] and ocp.us[i+1] based on the time
            # - calculate ocp.us[i] so that ocp.xs[i+1] = f(ocp.xs[i], ocp.us[i])
            #ocp.us[i] = ocp.us[i]

def create_ocp_velocity(
    rmodel: pin.Model, gmodel: pin.GeometryModel, param_parser: ParamParser
) -> crocoddyl.SolverAbstract:
    objects = {}

    # Stat and actuation model
    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFull(state)

    # Running & terminal cost models
    runningCostModel = crocoddyl.CostModelSum(state)
    terminalCostModel = crocoddyl.CostModelSum(state)

    ### Creation of cost terms

    # State Regularization cost
    xResidual = crocoddyl.ResidualModelState(state, param_parser.get_X0())
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)

    # Control Regularization cost
    uResidual = crocoddyl.ResidualModelControl(state)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)

    # End effector frame cost
    framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
        state,
        rmodel.getFrameId("panda2_hand_tcp"),
        param_parser.get_target_pose(),
    )
    objects["framePlacementResidual"] = framePlacementResidual

    goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)

    # Obstacle cost with hard constraint
    runningConstraintModelManager = crocoddyl.ConstraintModelManager(
        state, actuation.nu
    )
    terminalConstraintModelManager = crocoddyl.ConstraintModelManager(
        state, actuation.nu
    )
    # Creating the residual

    for col_idx, _ in enumerate(gmodel.collisionPairs):
        obstacleVelocityResidual = col.ResidualModelVelocityAvoidance(
            state,
            gmodel,
            col_idx,
            param_parser.get_di(),
            param_parser.get_ds(),
            param_parser.get_ksi(),
        )
        # Creating the inequality constraint
        constraint = crocoddyl.ConstraintModelResidual(
            state,
            obstacleVelocityResidual,
            np.array([0]),
            np.array([np.inf]),
        )

        # Adding the constraint to the constraint manager
        runningConstraintModelManager.addConstraint(f"col_{col_idx}", constraint)
        terminalConstraintModelManager.addConstraint(f"col_term_{col_idx}", constraint)

    # Adding costs to the models
    runningCostModel.addCost("stateReg", xRegCost, param_parser.get_W_xREG())
    runningCostModel.addCost("ctrlRegGrav", uRegCost, param_parser.get_W_uREG())
    runningCostModel.addCost(
        "gripperPoseRM", goalTrackingCost, param_parser.get_W_gripper_pose()
    )
    terminalCostModel.addCost("stateReg", xRegCost, param_parser.get_W_xREG())
    terminalCostModel.addCost(
        "gripperPose", goalTrackingCost, param_parser.get_W_gripper_pose_term()
    )

    # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
    running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state,
        actuation,
        runningCostModel,
        runningConstraintModelManager,
    )
    terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state,
        actuation,
        terminalCostModel,
        terminalConstraintModelManager,
    )

    runningModel = crocoddyl.IntegratedActionModelEuler(
        running_DAM, param_parser.get_dt()
    )
    terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.0)

    runningModel.differential.armature = np.full(7, 0.1)
    terminalModel.differential.armature = np.full(7, 0.1)

    problem = crocoddyl.ShootingProblem(
        param_parser.get_X0(), [runningModel] * param_parser.get_T(), terminalModel
    )
    # Create solver + callbacks
    # Define mim solver with inequalities constraints
    ocp = mim_solvers.SolverCSQP(problem)

    # Merit function
    ocp.use_filter_line_search = False

    # Parameters of the solver
    ocp.termination_tolerance = 1e-3
    ocp.max_qp_iters = 1000
    ocp.eps_abs = 1e-6
    ocp.eps_rel = 0

    ocp.with_callbacks = True

    return ocp, objects


def create_ocp_distance(
    rmodel: pin.Model,
    gmodel: pin.GeometryModel,
    use_distance_in_cost: bool,
    param_parser: ParamParser,
) -> crocoddyl.SolverAbstract:
    objects = {}

    # Stat and actuation model
    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFull(state)

    # Running & terminal cost models
    runningCostModel = crocoddyl.CostModelSum(state)
    terminalCostModel = crocoddyl.CostModelSum(state)

    ### Creation of cost terms

    # State Regularization cost
    xResidual = crocoddyl.ResidualModelState(state, param_parser.get_X0())
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)

    # Control Regularization cost
    uResidual = crocoddyl.ResidualModelControl(state)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)

    # End effector frame cost
    framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
        state,
        rmodel.getFrameId("panda2_hand_tcp"),
        param_parser.get_target_pose(),
    )
    objects["framePlacementResidual"] = framePlacementResidual

    goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)

    # Obstacle cost with hard constraint
    runningConstraintModelManager = crocoddyl.ConstraintModelManager(
        state, actuation.nu
    )
    terminalConstraintModelManager = crocoddyl.ConstraintModelManager(
        state, actuation.nu
    )
    # Creating the residual

    for col_idx, _ in enumerate(gmodel.collisionPairs):
        obstacleDistanceResidual = col.ResidualDistanceCollision(
            state, 7, gmodel, col_idx
        )
        # Creating the inequality constraint
        constraint = crocoddyl.ConstraintModelResidual(
            state,
            obstacleDistanceResidual,
            np.array([param_parser.get_safety_threshold()]),
            np.array([np.inf]),
        )

        # Adding the constraint to the constraint manager
        runningConstraintModelManager.addConstraint(f"col_{col_idx}", constraint)
        terminalConstraintModelManager.addConstraint(f"col_term_{col_idx}", constraint)

        if use_distance_in_cost:
            # Add the distance residual to the cost
            assert obstacleDistanceResidual.nr == 1
            activation = col.ActivationModelQuadExp(
                1, param_parser.get_distance_threshold() ** 2
            )
            cost = crocoddyl.CostModelResidual(
                state, activation, obstacleDistanceResidual
            )
            runningCostModel.addCost(
                f"col_{col_idx}", cost, param_parser.get_W_obstacle()
            )
            terminalCostModel.addCost(
                f"col_term_{col_idx}", cost, param_parser.get_W_obstacle()
            )

    # Adding costs to the models
    runningCostModel.addCost("stateReg", xRegCost, param_parser.get_W_xREG())
    runningCostModel.addCost("ctrlRegGrav", uRegCost, param_parser.get_W_uREG())
    runningCostModel.addCost(
        "gripperPoseRM", goalTrackingCost, param_parser.get_W_gripper_pose()
    )
    terminalCostModel.addCost("stateReg", xRegCost, param_parser.get_W_xREG())
    terminalCostModel.addCost(
        "gripperPose", goalTrackingCost, param_parser.get_W_gripper_pose_term()
    )

    # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
    running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state,
        actuation,
        runningCostModel,
        runningConstraintModelManager,
    )
    terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state,
        actuation,
        terminalCostModel,
        terminalConstraintModelManager,
    )

    runningModels = [ crocoddyl.IntegratedActionModelEuler(
        running_DAM, dt
    ) for dt in param_parser.get_dts() ]
    terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.0)

    for m in runningModels:
        m.differential.armature = np.full(7, 0.1)
    terminalModel.differential.armature = np.full(7, 0.1)

    problem = crocoddyl.ShootingProblem(
        param_parser.get_X0(), runningModels, terminalModel
    )
    # Create solver + callbacks
    # Define mim solver with inequalities constraints
    ocp = mim_solvers.SolverCSQP(problem)

    # Merit function
    ocp.use_filter_line_search = False

    # Parameters of the solver
    ocp.termination_tolerance = 1e-3
    ocp.max_qp_iters = 1000
    ocp.eps_abs = 1e-6
    ocp.eps_rel = 0

    ocp.with_callbacks = True

    return ocp, objects


def create_ocp_nocol(
    rmodel: pin.Model, param_parser: ParamParser
) -> crocoddyl.SolverAbstract:
    objects = {}

    # Stat and actuation model
    state = crocoddyl.StateMultibody(rmodel)
    actuation = crocoddyl.ActuationModelFull(state)

    ### Creation of cost terms

    # State Regularization cost
    xResidual = crocoddyl.ResidualModelState(state, param_parser.get_X0())
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)

    # Control Regularization cost
    uResidual = crocoddyl.ResidualModelControl(state)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)

    # End effector frame cost
    framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
        state,
        rmodel.getFrameId("panda2_hand_tcp"),
        param_parser.get_target_pose(),
    )
    objects["framePlacementResidual"] = framePlacementResidual

    goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)

    # Running & terminal cost models
    runningModels = []
    for dt in param_parser.get_dts():
        runningCostModel = crocoddyl.CostModelSum(state)
        runningCostModel.addCost("stateReg", xRegCost, param_parser.get_W_xREG())
        runningCostModel.addCost("ctrlRegGrav", uRegCost, param_parser.get_W_uREG())
        runningCostModel.addCost(
            "gripperPose", goalTrackingCost, param_parser.get_W_gripper_pose()
        )
        # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
        running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state,
            actuation,
            runningCostModel,
        )
        runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt)
        runningModel.differential.armature = np.full(7, 0.1)
        runningModels.append(runningModel)

    terminalCostModel = crocoddyl.CostModelSum(state)
    terminalCostModel.addCost("stateReg", xRegCost, param_parser.get_W_xREG())
    terminalCostModel.addCost(
        "gripperPose", goalTrackingCost, param_parser.get_W_gripper_pose_term()
    )

    # Create Differential Action Model (DAM), i.e. continuous dynamics and cost functions
    terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state,
        actuation,
        terminalCostModel,
    )

    terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.0)

    terminalModel.differential.armature = np.full(7, 0.1)

    problem = crocoddyl.ShootingProblem(
        param_parser.get_X0(), runningModels, terminalModel
    )
    # Create solver + callbacks
    # Define mim solver with inequalities constraints
    ocp = mim_solvers.SolverCSQP(problem)

    # Merit function
    ocp.use_filter_line_search = False

    # Parameters of the solver
    ocp.termination_tolerance = 1e-3
    ocp.max_qp_iters = 1000
    ocp.eps_abs = 1e-6
    ocp.eps_rel = 0

    ocp.with_callbacks = True

    return ocp, objects
