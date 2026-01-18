import random
import time

from collections import deque
import minari
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from scipy.spatial import KDTree

from planners.base_planner import Node, BasePlanner
from common.map_utils import create_local_map
from model.diffusion.conditional_unet1d import ConditionalUnet1D
from policies.fm_policy import DiffusionSampler


class RRT_Planner(BasePlanner):
    def __init__(self, start_state, goal_state, environment, sampler, **kwargs):
        super().__init__(start_state, goal_state, environment, sampler, **kwargs)

        self.kd_tree_dim = 2
        self.kd_tree = KDTree([start_state[:self.kd_tree_dim]])
        self.goal_sample_rate = 0.15
        self.goal_conditioning_bias = kwargs.get("goal_conditioning_bias", 0.85)  # default is 0.85
        self.prop_duration_schedule = kwargs.get("prop_duration", [64])  # default is [256,128,64] | [64]

    def reset(self):
        self.node_list.clear()
        self.node_list = [self.start_node]
        self.failed_node_list = []
        self.kd_tree = KDTree([self.start_node.state[:self.kd_tree_dim]])
        self.results = {"iterations": 0, "time": 0, "path": None, "actions": None, "number_of_nodes": 0}
        self.env.reset(options=self.options)

    def nearest_node(self, sample):
        _, index = self.kd_tree.query(sample[:, :self.kd_tree_dim], k=1)
        return self.node_list[index[0]]

    def nearest_node_batch(self, samples):
        _, index = self.kd_tree.query(samples)
        return [self.node_list[i] for i in index]
    
    def get_distance_to_goal(self, sample, discrete_path, dists_to_goal):
        """Returns the distance to goal and index on the discrete path for a given sample and the index on the discrete path."""
        dists = np.linalg.norm(discrete_path - sample[:2], axis=1)
        index = np.argmin(dists)
        return dists_to_goal[index], index
    
    def generate_intermediate_goals(self, start_node, goal_node, grid_size=20):
        """Generate intermediate goals using BFS on a grid map."""
        scale = self.s_global if hasattr(self, 's_global') else 1.0
        phys_width = self.map_width * scale
        phys_length = self.map_length * scale
        cell_w = phys_width / (2 * grid_size)
        cell_h = phys_length / (2 * grid_size)

        def map_to_cell(x, y):
            """Returns the cell coordinates (cx, cy) for a given map coordinate (x, y)."""
            cx = int(np.floor(x / cell_w))
            cy = int(np.floor(y / cell_h))
            return (cx, cy)

        def cell_to_center(cx, cy):
            """Returns the map coordinates (x, y) for the center of a given cell (cx, cy)."""
            x = (cx + 0.5) * cell_w
            y = (cy + 0.5) * cell_h
            if 'drone' in self.env_id.lower():
                return np.array([x, y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            elif 'ant' in self.env_id.lower():
                state = np.zeros(self.start_node.state.shape)
                state[0] = x; state[1] = y
                return state
            else:
                state = np.zeros(self.start_node.state.shape)
                state[0] = x; state[1] = y
                return state

        start_cell = map_to_cell(start_node[0], start_node[1])
        goal_cell  = map_to_cell(goal_node[0],  goal_node[1])
        
        range_min = -grid_size - 2
        range_max = grid_size + 2
        grid_cells = [(i, j) for i in range(range_min, range_max) for j in range(range_min, range_max)]
        
        # ------------------------------------------------------------
        # BFS TO FIND PATH ON GRID
        # ------------------------------------------------------------
        grid_centers = {}
        valid_cells = set()
        for c in grid_cells:
            state = cell_to_center(*c)
            try:
                if not self.check_collision(state):
                    grid_centers[c] = state
                    valid_cells.add(c)
            except IndexError: continue

        valid_cells.add(start_cell)
        valid_cells.add(goal_cell)

        queue = deque([start_cell])
        parent = {start_cell: None}
        visited = {start_cell}
        found = False
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            c = queue.popleft()
            if c == goal_cell:
                found = True
                break
            for dx, dy in directions:
                nxt = (c[0] + dx, c[1] + dy)
                if nxt in valid_cells and nxt not in visited:
                    visited.add(nxt)
                    parent[nxt] = c
                    queue.append(nxt)

        if not found: return [], [], []  # Retour vide x3

        # ------------------------------------------------------------
        # RECONSTRUCTION OF RAW PATH FROM BFS
        # ------------------------------------------------------------
        raw_path = []
        cur = goal_cell
        while cur is not None:
            if cur in grid_centers: raw_path.append(grid_centers[cur])
            else: raw_path.append(cell_to_center(*cur))
            cur = parent[cur]
        raw_path.reverse()

        raw_path[0] = start_node  # Ensure starting point is exact
        raw_path[-1] = goal_node  # Ensure goal point is exact
        
        if not raw_path: return [], [], [] # Retour vide x3

        # ------------------------------------------------------------
        # COMPUTE DISTANCE TO GOAL FOR EACH POINT ON RAW PATH
        # ------------------------------------------------------------
        dists_to_goal = np.zeros(len(raw_path))
        cumulative_dist = 0.0
        dists_to_goal[-1] = 0.0 # Last point is goal
        
        # Iterate backwards to compute cumulative distances
        for i in range(len(raw_path) - 2, -1, -1):
            p_curr = raw_path[i]
            p_next = raw_path[i+1]
            dist_segment = np.linalg.norm(p_next[:2] - p_curr[:2])
            
            cumulative_dist += dist_segment
            dists_to_goal[i] = cumulative_dist

        # ------------------------------------------------------------
        # SUB-SAMPLE THE RAW PATH BASED ON DESIRED SPACING
        # ------------------------------------------------------------
        
        sampled_path = []
        desired_spacing = self.max_v * self.env_dt * self.action_horizon * 10 
            
        current_segment_dist = 0.0

        for i in range(len(raw_path) - 1):
            p_curr = raw_path[i]
            p_next = raw_path[i+1]
            
            step_dist = np.linalg.norm(p_next[:2] - p_curr[:2])
            current_segment_dist += step_dist
            
            if current_segment_dist >= desired_spacing:
                sampled_path.append(p_next)
                current_segment_dist = 0.0 

        # Return the sampled intermediate goals, the raw path, and distances to goal
        return sampled_path, raw_path, dists_to_goal

    def plan(self):
        local_map = None
        start_time = time.time()
        curr_time = time.time()
        total_diffusion_time = 0
        i = 0
        
        #------------------------------------------------------------
        # PRE-COMPUTE INTERMEDIATE GOALS AND THEIR COSTS
        #------------------------------------------------------------
        grid_size = 20
        intermidiate_goal_states, discrete_path, dists_to_goal = self.generate_intermediate_goals(self.start_node.state,
                                                                                                  self.goal_state,
                                                                                                  grid_size)
        intermidiate_goal_states.append(self.goal_state)
        goals_arr = np.array([g[:2] for g in intermidiate_goal_states])
        path_arr = np.array([s[:2] for s in discrete_path])
        goals_costs = []
        for g in goals_arr:
            cost, _ = self.get_distance_to_goal(g, path_arr, dists_to_goal)
            goals_costs.append(cost)
        goals_costs = np.array(goals_costs)
        
        while (curr_time - start_time) < self.time_budget:
            if self.verbose:
                print(f"\rIteration: {i}, Elapsed Time: {(curr_time - start_time):.2f} seconds", end="")

            sample_node = self.random_node_sample()
            curr_node = self.nearest_node(sample_node)
            curr_state = curr_node.state
            # Now each node keeps track of which intermediate goal it is targeting
            current_goal_idx = curr_node.intermediate_goal_index
            full_action_seq = None
            full_states_seq = None
            done = False
            prev_actions = curr_node.parent_action_seq
            prev_states = curr_state[None, None, :] if curr_node.parent_states_seq is None \
                else curr_node.parent_states_seq  # (1,1,obs_dim)
            edge_length = self.prop_duration_schedule[curr_node.num_visit] if curr_node.num_visit < len(
                self.prop_duration_schedule) \
                else self.prop_duration_schedule[-1]
            # select random int value in range
            # edge_length = random.randint(self.prop_duration_schedule[0], self.prop_duration_schedule[1])
            curr_node.num_visit += 1
            
            for j in range(edge_length // self.action_horizon):  # default is 4
                
                #------------------------------------------------------------
                # UPDATE CURRENT INTERMEDIATE GOAL BASED ON DISTANCE TO GOAL
                #------------------------------------------------------------
                if current_goal_idx < len(goals_arr) - 1:
                    dist_to_goal, _ = self.get_distance_to_goal(curr_state, path_arr, dists_to_goal)
                    dist_to_next_goal_cost = goals_costs[current_goal_idx + 1]
                    
                    while dist_to_goal <= dist_to_next_goal_cost:
                        current_goal_idx += 1
                        if current_goal_idx >= len(goals_arr) - 1:
                            break
                        dist_to_next_goal_cost = goals_costs[current_goal_idx + 1]
                    
                    # Alternative criterion based on proximity
                    if current_goal_idx < len(goals_arr) - 1:
                        if np.linalg.norm(curr_state[:2] - goals_arr[current_goal_idx]) < 3 * self.max_v * self.env_dt * self.action_horizon:
                            current_goal_idx += 1
                
                if random.random() > self.goal_conditioning_bias:
                    # Random goal sampling
                    goal = sample_node[0, :2]
                else:
                    # Guided goal sampling
                    goal = intermidiate_goal_states[current_goal_idx][:2]
                    
                yaw = 0
                if "drone" in self.env_id.lower():
                    q = curr_state[6:10]
                    yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
                elif "car" in self.env_id.lower():
                    yaw = curr_state[2]
                local_map = create_local_map(self.maze, curr_state[0],
                                             curr_state[1], yaw,
                                             self.local_map_size,
                                             self.local_map_scale, self.s_global,
                                             (self.x_center, self.y_center))
                local_map = torch.tensor(local_map).to(self.device)
                # Sample and trim the action sequence
                start_diffusion_time = time.time()
                sampled_actions = self.sampler(prev_states, prev_actions=prev_actions,
                                               goal=goal, local_map=local_map)[0]
                end_diffusion_time = time.time()
                total_diffusion_time += (end_diffusion_time - start_diffusion_time)
                curr_state, done, curr_action_seq, curr_states_seq = self.propagate_action_sequence_env(curr_state,
                                                                                                        sampled_actions)

                # Fill edge's state/action sequence
                full_action_seq = (np.concatenate((full_action_seq, curr_action_seq))
                                   if full_action_seq is not None else curr_action_seq)
                prev_actions = curr_action_seq
                full_states_seq = (np.concatenate((full_states_seq, curr_states_seq), axis=1)
                                   if full_states_seq is not None else curr_states_seq)
                prev_states = curr_states_seq
                if done is None:  # Collision
                    if self.save_bad_edges:
                        self.failed_node_list.append(
                            Node(curr_state, full_action_seq, full_states_seq, parent=curr_node, intermediate_goal_index=current_goal_idx))
                    curr_state = None
                    break
                if done:
                    break

            if curr_state is not None:
                new_node = Node(curr_state, full_action_seq, full_states_seq, parent=curr_node, intermediate_goal_index=current_goal_idx)
                self.kd_tree = KDTree([node.state[:self.kd_tree_dim] for node in self.node_list])
                self.node_list.append(new_node)
                self.visualize_tree(filename=f'tree_plot/{i}')
            if done:
                # print diffusion time and total time
                curr_time = time.time()
                if self.verbose:
                    print(f"\rIteration: {i}, Elapsed Time: {(curr_time - start_time):.2f} seconds, "
                          f"Diffusion Time: {total_diffusion_time:.2f} seconds", end="")
                return self.handle_goal_reached(new_node, i, start_time)
                # if self.verbose:
                #     print(f" Goal reached in {i} iterations.")
                # path, actions = self.generate_final_path_env(new_node)
                # self.results["iterations"] = i
                # self.results["time"] = curr_time - start_time
                # self.results["path"] = path
                # self.results["path_time"] = len(path) * self.env_dt
                # self.results["actions"] = actions
                # self.results["number_of_nodes"] = len(self.node_list)
                # # self.visualize_tree()
                #
                # return path, actions

            i += 1
            curr_time = time.time()
        curr_time = time.time()
        if self.verbose:
            print(f"\rIteration: {i}, Elapsed Time: {(curr_time - start_time):.2f} seconds, "
                  f"Diffusion Time: {total_diffusion_time:.2f} seconds", end="")
        return self.handle_goal_not_reached(i, start_time)
        # if self.verbose:
        #     print(f" Goal not reached in {i} iterations.")
        # self.results["iterations"] = i
        # self.results["time"] = curr_time - start_time
        # self.results["number_of_nodes"] = len(self.node_list)
        # # self.visualize_tree()
        # return None, None


if __name__ == "__main__":
    # Settings
    debug = True
    time_budget = 120

    seed = 42
    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env_id = 'antmaze-large-diverse-v1'
    checkpoint = 'antmaze.pt'
    obs_horizon = 1
    num_diffusion_iters = 100
    if env_id == "antmaze-large-diverse-v1":
        obs_dim = 27
        action_dim = 8


    dataset = minari.load_dataset(env_id, download=False)
    render_mode = 'human' if debug else 'rgb_array'
    env = dataset.recover_environment(render_mode=render_mode)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = 'checkpoints/'
    checkpoint = torch.load(output_dir + checkpoint)
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon
    )
    noise_pred_net.load_state_dict(checkpoint['model_state_dict'])
    noise_pred_net = noise_pred_net.to(device)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    episode = dataset[2]
    start = episode.observations['observation'][0]
    start[2:] = 0  # start stationary
    goal = episode.observations['desired_goal'][1]

    diffusion_sampler = DiffusionSampler(noise_pred_net, noise_scheduler, env_id,
                                         pred_horizon=16,
                                         action_dim=2,
                                         obs_history=1,
                                         goal_conditioned=False
                                         )

    diffusion_planner = RRT_planner(start, goal,
                                    env_id=env_id,
                                    environment=env,
                                    sampler=diffusion_sampler,
                                    time_budget=time_budget,
                                    max_iter=300,
                                    verbose=True,
                                    render=True,
                                    )
    print("Planning with Diffusion Sampler...")
    path_diffusion, actions_diffusion = diffusion_planner.plan()

