import itertools
import threading
import time
import tkinter as tk
from tkinter import ttk
import typing as T

import crocoddyl
import numpy as np
import pinocchio
from param_parsers import ParamParser
from visualizer import add_sphere_to_viewer
from create_ocp import shift_result
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class GUI:
    def __init__(
        self,
        rmodel: pinocchio.Model,
        ref_frame_id: int,
        cmodel: pinocchio.GeometryModel,
        moving_geom: int,
        frame_placement_residual: crocoddyl.ResidualModelFramePlacement,
    ):
        self.rmodel = rmodel
        self.ref_frame_id = ref_frame_id
        self.cmodel = cmodel
        self.moving_geom = moving_geom
        self.frame_placement_residual = frame_placement_residual
        self.stop_requested = False
        self.pause = False

        self.mutex = threading.Lock()

    def update_geom(self):
        with self.mutex:
            self.cmodel.geometryObjects[
                self.moving_geom
            ].placement.translation = np.array(
                [
                    float(self.entry_geom_x.get()),
                    float(self.entry_geom_y.get()),
                    float(self.entry_geom_z.get()),
                ]
            )
            self.cmodel.geometryObjects[
                self.moving_geom
            ].placement.rotation = pinocchio.rpy.rpyToMatrix(
                float(self.entry_geom_roll.get()),
                float(self.entry_geom_pitch.get()),
                float(self.entry_geom_yaw.get()),
            )

    def update_reference(self):
        ref = pinocchio.SE3(
            pinocchio.rpy.rpyToMatrix(
                float(self.entry_ref_roll.get()),
                float(self.entry_ref_pitch.get()),
                float(self.entry_ref_yaw.get()),
            ),
            np.array(
                [
                    float(self.entry_ref_x.get()),
                    float(self.entry_ref_y.get()),
                    float(self.entry_ref_z.get()),
                ]
            ),
        )
        with self.mutex:
            self.frame_placement_residual.reference = ref
            self.rmodel.frames[self.ref_frame_id].placement = ref

            print(self.frame_placement_residual)

    def run(self):
        root = tk.Tk()
        root.title("3D Transformations")

        frame_geom = ttk.LabelFrame(root, text="Obstacle Transformation")
        frame_geom.grid(row=0, column=0, padx=10, pady=10)

        T = self.cmodel.geometryObjects[self.moving_geom].placement.translation
        ttk.Label(frame_geom, text="X:").grid(row=0, column=0)
        self.entry_geom_x = ttk.Entry(frame_geom)
        self.entry_geom_x.insert(0, str(T[0]))
        self.entry_geom_x.grid(row=0, column=1)

        ttk.Label(frame_geom, text="Y:").grid(row=1, column=0)
        self.entry_geom_y = ttk.Entry(frame_geom)
        self.entry_geom_y.insert(0, str(T[1]))
        self.entry_geom_y.grid(row=1, column=1)

        ttk.Label(frame_geom, text="Z:").grid(row=2, column=0)
        self.entry_geom_z = ttk.Entry(frame_geom)
        self.entry_geom_z.insert(0, str(T[2]))
        self.entry_geom_z.grid(row=2, column=1)

        R = self.cmodel.geometryObjects[self.moving_geom].placement.rotation
        rpy = pinocchio.rpy.matrixToRpy(R)
        ttk.Label(frame_geom, text="Roll:").grid(row=3, column=0)
        self.entry_geom_roll = ttk.Entry(frame_geom)
        self.entry_geom_roll.insert(0, str(rpy[0]))
        self.entry_geom_roll.grid(row=3, column=1)

        ttk.Label(frame_geom, text="Pitch:").grid(row=4, column=0)
        self.entry_geom_pitch = ttk.Entry(frame_geom)
        self.entry_geom_pitch.insert(0, str(rpy[1]))
        self.entry_geom_pitch.grid(row=4, column=1)

        ttk.Label(frame_geom, text="Yaw:").grid(row=5, column=0)
        self.entry_geom_yaw = ttk.Entry(frame_geom)
        self.entry_geom_yaw.insert(0, str(rpy[2]))
        self.entry_geom_yaw.grid(row=5, column=1)

        ttk.Button(frame_geom, text="Update Obstacle", command=self.update_geom).grid(
            row=6, column=1, padx=10, pady=10
        )

        frame_ref = ttk.LabelFrame(root, text="Reference Transformation")
        frame_ref.grid(row=1, column=0, padx=10, pady=10)

        T = self.frame_placement_residual.reference.translation
        ttk.Label(frame_ref, text="X:").grid(row=0, column=0)
        self.entry_ref_x = ttk.Entry(frame_ref)
        self.entry_ref_x.insert(0, str(T[0]))
        self.entry_ref_x.grid(row=0, column=1)

        ttk.Label(frame_ref, text="Y:").grid(row=1, column=0)
        self.entry_ref_y = ttk.Entry(frame_ref)
        self.entry_ref_y.insert(0, str(T[1]))
        self.entry_ref_y.grid(row=1, column=1)

        ttk.Label(frame_ref, text="Z:").grid(row=2, column=0)
        self.entry_ref_z = ttk.Entry(frame_ref)
        self.entry_ref_z.insert(0, str(T[2]))
        self.entry_ref_z.grid(row=2, column=1)

        R = self.frame_placement_residual.reference.rotation
        rpy = pinocchio.rpy.matrixToRpy(R)
        ttk.Label(frame_ref, text="Roll:").grid(row=3, column=0)
        self.entry_ref_roll = ttk.Entry(frame_ref)
        self.entry_ref_roll.insert(0, str(rpy[0]))
        self.entry_ref_roll.grid(row=3, column=1)

        ttk.Label(frame_ref, text="Pitch:").grid(row=4, column=0)
        self.entry_ref_pitch = ttk.Entry(frame_ref)
        self.entry_ref_pitch.insert(0, str(rpy[1]))
        self.entry_ref_pitch.grid(row=4, column=1)

        ttk.Label(frame_ref, text="Yaw:").grid(row=5, column=0)
        self.entry_ref_yaw = ttk.Entry(frame_ref)
        self.entry_ref_yaw.insert(0, str(rpy[2]))
        self.entry_ref_yaw.grid(row=5, column=1)

        ttk.Button(
            frame_ref, text="Update Reference", command=self.update_reference
        ).grid(row=6, column=1, padx=10, pady=10)

        frame_btn = ttk.LabelFrame(root, text="Simulation")
        frame_btn.grid(row=2, column=0, padx=10, pady=10)
        ttk.Button(frame_btn, text="Play/Pause", command=self.toggle_pause_request).grid(
            row=0, column=0, padx=10, pady=10
        )
        ttk.Button(frame_btn, text="Stop", command=self.stop_request).grid(
            row=0, column=1, padx=10, pady=10
        )
        root.wm_attributes("-topmost", True)

        self.root = root
        root.mainloop()

    def start(self):
        # Run the GUI in a separate thread
        self.gui_thread = threading.Thread(target=self.run)
        self.gui_thread.daemon = True
        self.gui_thread.start()

    def stop_request(self):
        self.stop_requested = True
        self.pause = False

    def toggle_pause_request(self):
        self.pause = not self.pause

    def stop(self):
        self.root.quit()
        self.gui_thread.join()

    def is_running(self):
        return self.gui_thread.is_alive()


class Simulation:
    def __init__(self, rmodel: crocoddyl.ActionModelAbstract, dt: float, x0: np.ndarray, nsteps: int = 1):
        self._rmodel = rmodel.copy()
        self._rdata = rmodel.createData()
        self._dt = dt
        self._dts = [ dt / nsteps ] * nsteps

        self._x = x0

        self._low_level_controller = self._default_controller

    def _default_controller(self, x, xdes, u, dt):
        return u

    def set_low_level_controller(self, func: T.Callable[[np.ndarray, np.ndarray, T.Optional[np.ndarray]], np.ndarray]):
        """The low level controller is a function that takes as parameters:
        - current state (q,v)
        - desired torque
        - time step
        - expected current state
        """
        self._low_level_controller = func

    def integrate_torque(self, u: np.ndarray, xs_expected: T.Optional[T.List[np.ndarray]] = None):
        # for dt in self._dts:
        #     # Calculate acceleration
        #     a = pinocchio.aba(self._rmodel, self._rdata, self._q, self._v, u)
        #     # integrate acceleration
        #     self._a = a
        #     self._v += a * dt
        #     dq = self._v * dt
        #     self._q = pinocchio.integrate(self._rmodel, self._q, dq)
        if xs_expected is None:
            xs_expected = [ None ] * (len(self._dts)+1)
        assert len(xs_expected) == len(self._dts)+1
        for dt, xdes in zip(self._dts, xs_expected):
            self._rmodel.dt = dt
            u_corrected = self._low_level_controller(self._x, xdes, u, dt)
            self._rmodel.calc(self._rdata, self._x, u_corrected)
            self._x = self._rdata.xnext

    def _interpolate(self, rmodel, rdata, x0, u):
        assert rmodel.dt == self._dt
        odt = rmodel.dt
        xs = [ x0.copy() ]
        # TODO: Use cumulative sum instead of the two loops below...
        for dt in self._dts:
            rmodel.dt = dt
            rmodel.calc(rdata, xs[-1], u)
            xs.append(rdata.xnext.copy())
        rmodel.dt = odt
        rmodel.calc(rdata, x0, u)
        dx = rdata.xnext - xs[-1]
        for i, t in enumerate(itertools.accumulate(self._dts)):
            xs[i+1] += dx * t / self._dt
        xs[-1] = rdata.xnext
        return xs

    def loop(self, ocp, xs, us, niters, iter_callback: T.Callable[[int], T.Any] = None):
        self._x = ocp.problem.x0
        states = [ self._x.copy() ]
        ocp_states = []
        ocp_controls = []
        for i in range(niters):
            ok = ocp.solve(xs, us, 100)
            ocp_states.append(ocp.xs.copy())
            ocp_controls.append(ocp.us.copy())
            # print(ok, ocp.iter)
            xs_expected = self._interpolate(ocp.problem.runningModels[0], ocp.problem.runningDatas[0], ocp.xs[0], ocp.us[0])
            self.integrate_torque(ocp.us[0], xs_expected)
            shift_result(ocp, self._dt)
            xs = ocp.xs.copy()
            us = ocp.us.copy()
            ocp.problem.x0 = self.x
            states.append(self.x.copy())
            if iter_callback is not None:
                iter_callback(i)
        return {
            "states": states,
            "ocp_states": ocp_states,
            "ocp_controls": ocp_controls,
        }

    @property
    def q(self) -> np.ndarray:
        return self._x[:self._rmodel.state.nq]

    @property
    def v(self) -> np.ndarray:
        return self._x[self._rmodel.state.nq:]
    
    @property
    def a(self) -> np.ndarray:
        return self._rdata.differential.xout

    @property
    def x(self) -> np.ndarray:
        return self._x

def simulation_loop(
    ocp: crocoddyl.SolverAbstract,
    rmodel: pinocchio.Model,
    ref_frame_id: int,
    cmodel: pinocchio.GeometryModel,
    moving_geom: int,
    frame_placement_residual: crocoddyl.ResidualModelFramePlacement,
    pp: ParamParser,
    vis: pinocchio.visualize.BaseVisualizer,
):
    rdata = pinocchio.Data(rmodel)
    vis_id = frame_placement_residual.id  # rmodel.getFrameId("panda2_rightfinger")

    gui = GUI(rmodel, ref_frame_id, cmodel, moving_geom, frame_placement_residual)
    gui.start()
    gui.toggle_pause_request()

    simulation = Simulation(ocp.problem.runningModels[0], pp.get_dt(), pp.get_X0(), 100)
    np.set_printoptions(linewidth=200)

    try:
        dt = pp.get_dt()

        xs = [pp.get_X0()] * (pp.get_T() + 1)
        us = ocp.problem.quasiStatic(xs[:-1])

        ok = ocp.solve(xs, us, 1000)
        print(ok, ocp.cost)
        assert ok
        xs = ocp.xs.copy()
        us = ocp.us.copy()

        i = 0
        while not gui.stop_requested:
            while gui.pause:
                time.sleep(dt)
            i = i + 1
            t0 = time.time()
            with gui.mutex:
                ok = ocp.solve(xs, us, 100)
            t1 = time.time()
            xs = ocp.xs.copy()
            us = ocp.us.copy()
            
            simulation.integrate_torque(us[0])
            if i % 10 == 0:
                print("    i ok iter  time")
            cbegin = ""
            cend = ""
            if not ok or (t1-t0) >= dt:
                cbegin = bcolors.FAIL
                cend = bcolors.ENDC
            print(f"{cbegin}{i:5}  {ok:1}  {ocp.iter:3}  {(t1-t0)*1e3:4.2f}{cend}")
            
            # Shift the trajectory
            shift_result(ocp, dt)
            ocp.problem.x0 = simulation.x

            # Update the visualizer
            vis.display(simulation.q)

            for k, x in enumerate(ocp.xs):
                qq = x[:rmodel.nq]
                pinocchio.forwardKinematics(rmodel, rdata, qq)
                pinocchio.updateFramePlacement(rmodel, rdata, vis_id)
                add_sphere_to_viewer(
                    vis,
                    f"colmpc{k}",
                    2e-2,
                    rdata.oMf[vis_id].translation,
                    color=100000,
                )

            sleep_time = dt - (t1 - t0)
            time.sleep(max(0, sleep_time))
    finally:
        gui.stop()
