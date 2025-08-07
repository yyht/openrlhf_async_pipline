import pickle
from os import PathLike
import pickle
from os import PathLike
import math
from copy import deepcopy
from typing import Generic, Optional, NamedTuple, Callable, Hashable
import itertools
from abc import ABC
from abc import ABC
from collections import defaultdict

import numpy as np
import random
from tqdm import trange

import sys
sys.setrecursionlimit(1500)

import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import sys, os
from openrlhf.env.base import SearchAlgorithm, WorldModel, SearchConfig, State, Action, Example, Trace

class MCTSNode(Generic[State, Action, Example]):
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, state: Optional[State], action: Optional[Action], 
                 parent: "Optional[MCTSNode]" = None,
                 fast_reward: float = 0., fast_reward_details=None,
                 is_terminal: bool = False, 
                 calc_q: Callable[[list[float]], float] = np.mean,
                 reward=0.0,
                 value=0.0
                ):
        """
        A node in the MCTS search tree

        :param state: the current state
        :param action: the action of the last step, i.e., the action from parent node to current node
        :param parent: the parent node, None if root of the tree
        :param fast_reward: an estimation of the reward of the last step
        :param is_terminal: whether the current state is a terminal state
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        """
        self.id = next(MCTSNode.id_iter)
        if fast_reward_details is None:
            fast_reward_details = {}
        self.cum_rewards: list[float] = []
        self.fast_reward = self.reward = fast_reward
        self.fast_reward_details = fast_reward_details
        self.is_terminal = is_terminal
        self.action = action
        self.state = state
        self.parent = parent
        self.value = value
        self.children: 'Optional[list[MCTSNode]]' = None
        self.calc_q = calc_q
        self.reward = reward
        self.N = 0
        self.V = 0.0
        self.Q = self.parent.V + self.reward if self.parent is not None else self.reward
        
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
        if self.parent:
            self.parent_id = self.parent.id
        else:
            self.parent_id = self.id
            
    @property
    def info(self):
        info = {
            'is_terminal': self.is_terminal,
            'N': self.N,
            'V': self.V,
            'Q': self.Q,
            'Q_hat': self.Q_hat,
            'action': self.action,
            'reward': self.reward,
            'cum_rewards': self.cum_rewards,
            'id': self.id,
            'parent_id': self.parent_id,
            'depth': self.depth
        }
        return info
    
    @property
    def r(self) -> float:
        if self.rewards is None:
            return self.value if self.parent is None else (self.value - self.parent.value)
        # TODO: consider KL divergence in MCTS
        # return self.rewards.mean().detach().item() + (self.value if self.parent is None else (self.value - self.parent.value))
        raise ValueError('Should not consider kl divergence here!')

    # noinspection PyPep8Naming
    @property
    def Q_hat(self) -> float:
        if self.state is None:
            return self.fast_reward
        else:
            return self.calc_q(self.cum_rewards)


class MCTSResult(NamedTuple):
    terminal_state: State
    cum_reward: float
    trace: Trace
    trace_of_nodes: list[MCTSNode]
    tree_state: MCTSNode
    trace_in_each_iter: list[list[MCTSNode]] = None
    tree_state_after_each_iter: list[MCTSNode] = None
    aggregated_result: Optional[Hashable] = None


class MCTSAggregation(Generic[State, Action, Example], ABC):
    def __init__(self, retrieve_answer: Callable[[State], Hashable],
                 weight_policy: str = 'edge'):
        assert weight_policy in ['edge', 'edge_inverse_depth', 'uniform']
        self.retrieve_answer = retrieve_answer
        self.weight_policy = weight_policy

    def __call__(self, tree_state: MCTSNode[State, Action,Example]) -> Optional[Hashable]:
        answer_dict = defaultdict(lambda: 0)

        def visit(cur: MCTSNode[State, Action, Example]):
            if cur.state is None:
                return []
            if cur.is_terminal:
                answer = self.retrieve_answer(cur.state)
                if answer is None:
                    print("MCTSAggregation: no answer retrieved.")
                    return []
                if self.weight_policy == 'edge':
                    answer_dict[answer] += cur.reward
                elif self.weight_policy == 'edge_inverse_depth':
                    answer_dict[answer] += cur.reward / cur.depth
                elif self.weight_policy == 'uniform':
                    answer_dict[answer] += 1.0
                return [(answer, cur.depth)]
            depth_list = defaultdict(list)
            cur_list = []
            for child in cur.children:
                cur_list.extend(child_info := visit(child))
                for answer, depth in child_info:
                    depth_list[answer].append(depth)
            for answer, depths in depth_list.items():
                if self.weight_policy == 'edge':
                    answer_dict[answer] += cur.reward
                elif self.weight_policy == 'edge_inverse_depth':
                    answer_dict[answer] += cur.reward / np.mean(depths)
            return cur_list

        visit(tree_state)

        if len(answer_dict) == 0:
            return None
        return max(answer_dict, key=lambda answer: answer_dict[answer])

class MCTS(SearchAlgorithm, Generic[State, Action, Example]):
    def __init__(self,
                 gamma=0.9,
                 output_trace_in_each_iter: bool = False,
                 w_exp: float = 1.,
                 depth_limit: int = 5,
                 n_iters: int = 10,
                 cum_reward: Callable[[list[float]], float] = sum,
                 calc_q: Callable[[list[float]], float] = np.mean,
                 simulate_strategy: str | Callable[[list[float]], int] = 'max',
                 output_strategy: str = 'max_reward',
                 uct_with_fast_reward: bool = True,
                 aggregator: Optional[MCTSAggregation] = None,
                 disable_tqdm: bool = True,
                 node_visualizer: Callable[[MCTSNode], dict] = lambda x: x.__dict__):
        """
        MCTS algorithm

        :param output_trace_in_each_iter: whether to output the trace of the chosen trajectory in each iteration ; the trace is *deepcopy*-ed
                                          will also output *tree_state_after_each_iter*, which is the *deepcopy*-ed root
        :param w_exp: the weight of exploration in UCT
        :param cum_reward: the way to calculate the cumulative reward from each step. Defaults: sum
        :param calc_q: the way to calculate the Q value from histories. Defaults: np.mean
        :param simulate_strategy: simulate strategy. Options: 'max', 'sample', 'random', or use a custom function
        :param output_strategy: the way to output the result. The nodes are not *deepcopy*-ed, so the information is after all iterations
                                Options: 'max_reward': dfs on the final tree to find a trajectory with max reward using :param cum_reward:
                                         'follow_max': starting from root, choose the maximum reward child at each step. May output a non-terminal node if dead end
                                         'max_visit': the terminal node with maximum number of visits
                                         'max_iter': the trajectory with a terminal node and max reward among those in each iteration
                                         'last_iter': the last trajectory. May output a non-terminal node if the last iteration leads to a dead end
                                         'last_terminal_iter': the last trajectory with a terminal node
                                Outputs *None* if no trajectory with terminal node but required
        :param uct_with_fast_reward: if True, use fast_reward instead of reward for unvisited children in UCT
                                     Otherwise, visit the *unvisited* children with maximum fast_reward first
        """
        super().__init__()
        self.gamma = gamma
        self.world_model = None
        self.search_config = None
        self.output_trace_in_each_iter = output_trace_in_each_iter
        self.w_exp = w_exp
        self.depth_limit = depth_limit
        self.n_iters = n_iters
        self.cum_reward = cum_reward
        self.calc_q = calc_q
        default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            'max': lambda x: np.argmax(x),
            'sample': lambda x: np.random.choice(len(x), p=x),
            'random': lambda x: np.random.choice(len(x)),
        }
        self.simulate_choice: Callable[[list[float]], int] = default_simulate_strategies.get(simulate_strategy,
                                                                                             simulate_strategy)
        assert output_strategy in ['max_reward', 'follow_max', 'max_visit', 'max_iter', 'last_iter',
                                   'last_terminal_iter', 'max_q']
        self.output_strategy = output_strategy
        self.uct_with_fast_reward = uct_with_fast_reward
        self._output_iter: list[MCTSNode] = None
        self._output_cum_reward = -math.inf
        self.trace_in_each_iter: list[list[MCTSNode]] = None
        self.root: Optional[MCTSNode] = None
        self.disable_tqdm = disable_tqdm
        self.node_visualizer = node_visualizer
        self.aggregator = aggregator
        self.node_visualizer = node_visualizer
        self.aggregator = aggregator
        
    def iterate(self, node: MCTSNode) -> list[MCTSNode]:
        path = self._select(node)
        while not self._is_terminal_with_depth_limit(path[-1]):
            self._expand(path[-1])
            if self._is_terminal_with_depth_limit(path[-1]) or len(path[-1].children) == 0:
                break
            node = self._uct_select(path[-1])
            path.append(node)
        self._back_propagate(path)
        return path

#     def iterate(self, node: MCTSNode) -> list[MCTSNode]:
#         path = self._select(node)
#         if not self._is_terminal_with_depth_limit(path[-1]):
#             self._expand(path[-1])
#             self._simulate(path)
#         cum_reward = self._back_propagate(path)
        
#         return path
    
    def _get_simulated_pi(self, cur_node: MCTSNode, return_selection=False) -> list[float]:
        """
        Apated from: https://github.com/suragnair/alpha-zero-general/blob/ce020c8eebbabf0e22654279508a6887b4791015/MCTS.py#L28C5-L53C21
        """
        visit_counts = [child.N for child in cur_node.children]
        next_action_V = [child.V for child in cur_node.children]
        next_action_Q = [child.Q for child in cur_node.children]
        next_action_n_children = [len(child.children) if child.children is not None else 0 for child in cur_node.children]
        next_action_variance = [calculate_diversity_score(child.children) for child in cur_node.children]
        
        def _cal_probs(temp):
            if temp > 0:
                try:
                    ## choice 1: to sample based on visit counts
                    # counts = [(x * (nc + 1 if self.consider_diversity else 1)) ** (1. / temp) if x else x \
                    #     for x, nc in zip(visit_counts, next_action_n_children)]
                    ## choice 2: to sample based on Q values
                    counts = [(math.exp(x) * (nc + 1 if self.consider_diversity else 1)) ** (1. / temp) if x else x \
                        for x, nc in zip(next_action_Q, next_action_n_children)]
                    total_count = float(sum(counts))
                    probs = [x / total_count for x in counts]
                    return probs
                except OverflowError as e:
                    print(('Run into {} -- Temperature too small ... Set to zero ...').format(str(e)))
            best_actions = np.array(np.argwhere(visit_counts == np.max(visit_counts))).flatten()
            probs = [0] * len(visit_counts)
            for best_action in best_actions:
                probs[best_action] = 1 / len(best_actions)
            return probs
        
        temperature = self.temperature * (self.temperature_decay_ratio ** cur_node.depth)
        probs = _cal_probs(temperature)
        
        if return_selection:
            if temperature == 0:
                ## choice 1: to sample based on visit counts
                # selected_idx = max(range(len(visit_counts)), key=lambda x: (
                #     (next_action_Q[x] + 2) * (next_action_variance[x] + 1 if self.consider_diversity else 1), 
                #     visit_counts[x], next_action_V[x]
                # ))
                ## choice 2: to sample based on Q values
                selected_idx = max(range(len(visit_counts)), key=lambda x: (
                    visit_counts[x] * (next_action_variance[x] + 1 if self.consider_diversity else 1), 
                    next_action_Q[x], next_action_V[x]
                ))
            else:
                selected_idx = np.random.choice(range(len(visit_counts)), p=probs)
            return probs, selected_idx, next_action_V, next_action_Q
        return probs, next_action_V, next_action_Q

    def _is_terminal_with_depth_limit(self, node: MCTSNode):
        return node.is_terminal or node.depth >= self.depth_limit

    def _select(self, node: MCTSNode) -> list[MCTSNode]:
        path = []
        while True:
            path.append(node)
            if node.children is None or len(node.children) == 0 or self._is_terminal_with_depth_limit(node):
                return path
            node = self._uct_select(node)
            
#     def _uct(self, node: MCTSNode) -> float:
#         return node.Q + self.w_exp * np.sqrt(np.log(len(node.parent.cum_rewards)) / max(1, len(node.cum_rewards)))

#     def _uct_select(self, node: MCTSNode) -> MCTSNode:
#         if self.uct_with_fast_reward or all(x.state is not None for x in node.children):
#             return max(node.children, key=self._uct)
#         else:
#             unvisited_children = filter(lambda x: x.state is None, node.children)
#             return max(unvisited_children, key=lambda x: x.fast_reward)

    # use uct
    # https://github.com/YuxiXie/MCTS-DPO/blob/main/mcts_rl/algorithms/mcts/mcts/mcts.py
    def _uct(self, node: MCTSNode) -> float:
        return node.Q + self.w_exp * np.sqrt(np.log(node.parent.N) / max(1, node.N))

    # use uct to select a node
    def _uct_select(self, node: MCTSNode) -> MCTSNode:
        logger.info('###进入select###')
        # if self.uct_with_fast_reward or all(x.state is not None for x in node.children) or random.random() < 0.5:
        #     logger.info('###full-child-uct-离开select###')
        #     return max(node.children, key=self._uct)
        if self.uct_with_fast_reward or all(x.state is not None for x in node.children):
            logger.info('###full-child-uct-离开select###')
            return max(node.children, key=self._uct)
        else:
            # first explore unvisited_children
            unvisited_children = [child for child in node.children if child.state is None]
            fast_rewards = [child.fast_reward for child in unvisited_children]
            logger.info('###explore unvisited_children-离开select###')
            return unvisited_children[self.simulate_choice(fast_rewards)]

    def _expand(self, node: MCTSNode):
        logger.info('###进入expand###')
        if node.state is None: 
            node.state, aux = self.world_model.step(node.parent.state, node.action)
            # reward is calculated after the state is updated, so that the
            # information can be cached and passed from the world model
            # to the reward function with **aux without repetitive computation
            node.reward, node.reward_details = self.search_config. \
                reward(node.parent.state, node.action, **node.fast_reward_details, **aux)
            # update initial q for each node
            node.Q = node.parent.V + node.reward if node.parent is not None else node.reward
            node.is_terminal = self.world_model.is_terminal(node.state)

        if node.is_terminal:
            return

        children = []
        actions = self.search_config.get_actions(node.state)
        for action in actions:
            fast_reward, fast_reward_details = self.search_config.fast_reward(node.state, action)
            child = MCTSNode(state=None, action=action, parent=node,
                             fast_reward=fast_reward, fast_reward_details=fast_reward_details, 
                             calc_q=self.calc_q)
            children.append(child)
        # node.children = children
        node.children = children if node.children is None else node.children + children
        logger.info('###离开expand###')

    def _simulate(self, path: list[MCTSNode]):
        logger.info('###进入simulate###')
        node = path[-1]
        while True:
            if node.state is None:
                self._expand(node)
            if self._is_terminal_with_depth_limit(node) or len(node.children) == 0:
                return
            fast_rewards = [child.fast_reward for child in node.children]
            # fast_rewards = [child.Q for child in node.children]
            node = node.children[self.simulate_choice(fast_rewards)]
            path.append(node)
        logger.info('###离开simulate###')

    # def _back_propagate(self, path: list[MCTSNode]):
    #     rewards = []
    #     cum_reward = -math.inf
    #     for node in reversed(path):
    #         rewards.append(node.reward)
    #         cum_reward = self.cum_reward(rewards[::-1])
    #         node.cum_rewards.append(cum_reward)
    #     return cum_reward
    
    ## adapted from https://github.com/YuxiXie/MCTS-DPO
    def _back_propagate(self, path: list[MCTSNode]):
        logger.info('###进入back_propagate###')
        node = path[-1]
        node.Q = node.reward + self.gamma * node.V
        node.N += 1
        for node in reversed(path[:-1]):
            node.V = sum(max(1, child.N) * child.Q for child in node.children) / sum(max(1, child.N) for child in node.children)
            node.N += 1
            if node.action is not None:
                node.Q = node.reward + self.gamma * node.V
             
        rewards = []
        cum_reward = -math.inf
        for node in reversed(path):
            # logger.info('###back_propagate-node-before###')
            # logger.info(node.info)
            rewards.append(node.reward)
            cum_reward = self.cum_reward(rewards[::-1])
            node.cum_rewards.append(cum_reward)
            # logger.info('###back_propagate-node-after###')
            # logger.info(node.info)
        logger.info('###离开back_propagate###')
        return cum_reward

    def _dfs_max_reward(self, path: list[MCTSNode]) -> tuple[float, list[MCTSNode]]:
        cur = path[-1]
        if cur.is_terminal:
            return self.cum_reward([node.Q for node in path[1:]]), path
        if cur.children is None:
            return -math.inf, path
        visited_children = [x for x in cur.children if x.state is not None]
        if len(visited_children) == 0:
            return -math.inf, path
        return max((self._dfs_max_reward(path + [child]) for child in visited_children), key=lambda x: x[0])
    
    def output_trace(self, node):
        output_trace = {}
        for strategy in ['max_reward', 'max_q', 'max_visit']:
            output_trace[strategy] = {
                'trace': [],
                'cum_reward': 0.0
            }
            _output_iter = []
            if strategy == 'max_reward':
                _, _output_iter = self._dfs_max_reward([node])
            elif strategy == 'max_q':
                _output_iter = []
                cur = node
                while True:
                    _output_iter.append(cur)
                    if cur.is_terminal:
                        break
                    visited_children = [x for x in cur.children if x.state is not None]
                    if len(visited_children) == 0:
                        break
                    cur = max(visited_children, key=lambda x: x.Q)
            elif strategy == 'max_visit':
                _output_iter = []
                cur = node
                while True:
                    _output_iter.append(cur)
                    if cur.is_terminal:
                        break
                    visited_children = [x for x in cur.children if x.state is not None]
                    if len(visited_children) == 0:
                        break
                    cur = max(visited_children, key=lambda x: x.N)
            else:
                continue
            if _output_iter:
                output_trace[strategy]['trace'] = _output_iter[1:]
                output_trace[strategy]['cum_reward'] = self.cum_reward([node.Q for node in _output_iter[1::]])
        return output_trace
                    
    def search(self):
        self._output_cum_reward = -math.inf
        self._output_iter = None
        self.root = MCTSNode(state=self.world_model.init_state(), action=None, parent=None, 
                             calc_q=self.calc_q)
        if self.output_trace_in_each_iter:
            self.trace_in_each_iter = []

        for _ in trange(self.n_iters, disable=self.disable_tqdm, desc='MCTS iteration', leave=False):
            path = self.iterate(self.root)
            # logger.info(self.root.info)
            if self.output_trace_in_each_iter:
                # self.trace_in_each_iter.append(deepcopy(path))
                self.trace_in_each_iter.append(path)

    def __call__(self,
                 world_model: WorldModel[State, Action, Example],
                 search_config: SearchConfig[State, Action, Example],
                 log_file: Optional[str] = None,
                 **kwargs) -> MCTSResult:
        MCTSNode.reset_id()
        self.world_model = world_model
        self.search_config = search_config

        self.search()

        if self._output_iter is None:
            terminal_state = trace = None
        else:
            terminal_state = self._output_iter[-1].state
            trace = [node.state for node in self._output_iter], [node.action for node in self._output_iter[1:]]
        if self.output_trace_in_each_iter:
            trace_in_each_iter = self.trace_in_each_iter
            tree_state_after_each_iter = [trace[0] for trace in trace_in_each_iter]
        else:
            trace_in_each_iter = tree_state_after_each_iter = None
        result = MCTSResult(terminal_state=terminal_state,
                            cum_reward=self._output_cum_reward,
                            trace=trace,
                            trace_of_nodes=self._output_iter,
                            tree_state=self.root,
                            trace_in_each_iter=trace_in_each_iter,
                            tree_state_after_each_iter=tree_state_after_each_iter)
        if self.aggregator is not None:
            result = MCTSResult(
                terminal_state=result.terminal_state,
                cum_reward=result.cum_reward,
                trace=result.trace,
                trace_of_nodes=result.trace_of_nodes,
                tree_state=result.tree_state,
                trace_in_each_iter=result.trace_in_each_iter,
                tree_state_after_each_iter=result.tree_state_after_each_iter,
                aggregated_result=self.aggregator(result.tree_state),
            )
        return result