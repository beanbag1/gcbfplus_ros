# import sys
# sys.path.append('/root/ros_ws/src/gcbfplus_ros/src')
# sys.path.insert(0, '/usr/lib/python3.10')
# print(sys.path)

import argparse
import datetime
import functools as ft
import os
import pathlib
import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import yaml

from gcbfplus.algo import GCBF, GCBFPlus, make_algo, CentralizedCBF, DecShareCBF
from gcbfplus.env import make_env
from gcbfplus.env.base import RolloutResult
from gcbfplus.trainer.utils import get_bb_cbf
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import jax_jit_np, tree_index, chunk_vmap, merge01, jax_vmap, jax2np, np2jax

import rospy
import globals
from ros_interface import init_ros_interface, send_control
import open3d as o3d


def simulate(args):
    print(f"> Running simulate.py {args}")

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")

    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)
    np.random.seed(args.seed)

    # load config
    if not args.u_ref and args.path is not None:
        with open(os.path.join(args.path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)
            print("loaded")

    # create environments
    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=num_agents,
        num_obs=args.obs,
        area_size=args.area_size,
        max_step=args.max_step,
        max_travel=args.max_travel,
    )

    if not args.u_ref:
        if args.path is not None:
            path = args.path
            model_path = os.path.join(path, "models")
            if args.step is None:
                models = os.listdir(model_path)
                step = max([int(model) for model in models if model.isdigit()])
            else:
                step = args.step
            print("step: ", step)

            algo = make_algo(
                algo=config.algo,
                env=env,
                node_dim=env.node_dim,
                edge_dim=env.edge_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                n_agents=env.num_agents,
                gnn_layers=config.gnn_layers,
                batch_size=config.batch_size,
                buffer_size=config.buffer_size,
                horizon=config.horizon,
                lr_actor=config.lr_actor,
                lr_cbf=config.lr_cbf,
                alpha=config.alpha,
                eps=0.02,
                inner_epoch=8,
                loss_action_coef=config.loss_action_coef,
                loss_unsafe_coef=config.loss_unsafe_coef,
                loss_safe_coef=config.loss_safe_coef,
                loss_h_dot_coef=config.loss_h_dot_coef,
                max_grad_norm=2.0,
                seed=config.seed
            )
            algo.load(model_path, step)
            act_fn = jax.jit(algo.act)
        else:
            algo = make_algo(
                algo=args.algo,
                env=env,
                node_dim=env.node_dim,
                edge_dim=env.edge_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                n_agents=env.num_agents,
                alpha=args.alpha,
            )
            act_fn = jax.jit(algo.act)
            path = os.path.join(f"./logs/{args.env}/{args.algo}")
            if not os.path.exists(path):
                os.makedirs(path)
            step = None
    else:
        assert args.env is not None
        path = os.path.join(f"./logs/{args.env}/nominal")
        if not os.path.exists("./logs"):
            os.mkdir("./logs")
        if not os.path.exists(os.path.join("./logs", args.env)):
            os.mkdir(os.path.join("./logs", args.env))
        if not os.path.exists(path):
            os.mkdir(path)
        algo = None
        act_fn = jax.jit(env.u_ref)
        step = 0

    algo_is_cbf = isinstance(algo, (CentralizedCBF, DecShareCBF))

    if args.cbf is not None:
        assert isinstance(algo, GCBF) or isinstance(algo, GCBFPlus) or isinstance(algo, CentralizedCBF)
        get_bb_cbf_fn_ = ft.partial(get_bb_cbf, algo.get_cbf, env, agent_id=args.cbf, x_dim=0, y_dim=1)
        get_bb_cbf_fn_ = jax_jit_np(get_bb_cbf_fn_)

        def get_bb_cbf_fn(T_graph: GraphsTuple):
            T = len(T_graph.states)
            outs = [get_bb_cbf_fn_(tree_index(T_graph, kk)) for kk in range(T)]
            Tb_x, Tb_y, Tbb_h = jtu.tree_map(lambda *x: jnp.stack(list(x), axis=0), *outs)
            return Tb_x, Tb_y, Tbb_h
    else:
        get_bb_cbf_fn = None
        cbf_fn = None

    def body(graph):
        action = act_fn(graph)
        return action

    jit_body = jax.jit(body)
    jit_graph = jax.jit(env.get_graph)
    
    test_key = jr.PRNGKey(args.seed)

    factor = 8
    # anyhow choose a number that seems to work

    goal = globals.goal
    goal = goal/factor
    states = globals.odom_data

    # placeholder obstacles so that the function can run
    obstacle_info = [jr.uniform(test_key, (0, 2)), jnp.array([]), jnp.array([]), jnp.array([]), jnp.array([])]
    obstacles = env.create_obstacles(obstacle_info[0], obstacle_info[1], obstacle_info[2], obstacle_info[3])
    # obstacles = env.create_obstacles(obstacle_info[0], obstacle_info[1])

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # point_cloud = o3d.geometry.PointCloud()
    # first_geom = True

    while not rospy.is_shutdown():
        globals.start_time = rospy.get_time()
        start_time = rospy.get_time()
        while not globals.lidar_lock: 
            rospy.sleep(0.005)
        time_elapsed = (rospy.get_time() - start_time)*1000
        # print(f"time elapsed: {time_elapsed}ms")
        globals.lidar_data = np2jax(globals.new_lidar_data[:, :2])
        lidar_data = globals.lidar_data/factor
        lol = globals.new_lidar_data
        # print(globals.lidar_data)
        globals.num_rays = len(globals.lidar_data)
        # print(globals.num_rays)

        goal = globals.goal
        goal = goal/factor
        odom_data = globals.odom_data/factor
        vel_data = globals.vel_data/factor
        states = jnp.concatenate([odom_data[:, :2], vel_data[:, :2]], axis = -1)
        print(f"states: {states}")
        env.n_rays = globals.num_rays
        env.env_states = env.EnvState(states, goal, obstacles)

        time_elapsed = (rospy.get_time() - start_time)*1000
        # print(f"time elapsed: {time_elapsed}ms")
        
        # apply control
        graph = jit_graph(env.env_states, lidar_data)
        # graph = env.get_graph(env.env_states, globals.lidar_data)

        time_elapsed = (rospy.get_time() - start_time)*1000
        # print(f"time elapsed: {time_elapsed}ms")

        action = jax2np(jit_body(graph))
        action = env.agent_accel(action)
        # action = algo.act(graph)

        print(f"action: {action}")
        # globals.control_vector = action[0]
        globals.lidar_lock = False
        time_elapsed = (rospy.get_time() - start_time)*1000
        print(f"FINAL time elapsed: {time_elapsed}ms")

        send_control(action[0], vel_data[0], time_elapsed/1000)

        error = goal[:, :2] - odom_data[:, :2]
        dist = np.sqrt(error[:, 0]**2 + error[:, 1]**2)
        if dist < 0.1:
            print("goal reached")

        # rm_list = np.empty(1, dtype=int)
        # for i in range(len(lol)):
        #     temp = lol[i]
        #     dist = np.sqrt(temp[0]**2 + temp[1]**2)
        #     if dist > 3.0:
        #         rm_list = np.append(rm_list, i)
        # rm_list = np.delete(rm_list, 0)
        # lol = np.delete(lol, rm_list)

        # point_cloud.points = o3d.utility.Vector3dVector(np.append(lol, globals.odom_data, axis=0))
        # # point_cloud.points = o3d.utility.Vector3dVector(graph.edges[:, :3])
        # if first_geom:
        #     vis.add_geometry(point_cloud)
        #     first_geom = False
        # else:
        #     vis.remove_geometry(point_cloud)
        #     vis.add_geometry(point_cloud)
        # vis.poll_events()
        # vis.update_renderer()



def main():
    init_ros_interface()

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-agents", type=int, default=1)
    parser.add_argument("--obs", type=int, default=0)
    parser.add_argument("--area-size", type=float, default=50)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-travel", type=float, default=None)
    parser.add_argument("--cbf", type=int, default=None)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--u-ref", action="store_true", default=False)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--epi", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--nojit-rollout", action="store_true", default=False)
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--dpi", type=int, default=100)

    args = parser.parse_args()
    simulate(args)

    rospy.spin()

if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
