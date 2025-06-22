import time
import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.base import Chain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.llms.base import BaseLLM

from langchain.requests import RequestsWrapper

from .planner import Planner
from .api_selector import APISelector
from .caller import Caller
from utils import ReducedOpenAPISpec
from .myutil import Dataloader
import io
import json
import pandas as pd
from causallearn.utils.cit import CIT
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI
import os
from matplotlib import image as mpimg, pyplot as plt
import warnings
from langchain.agents import Tool
import matplotlib
from econml.dml import LinearDML
from .myutil import has_confounder, has_collider, Relationship, Dataloader, CGMemory
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
logger = logging.getLogger(__name__)

folder = './DGP_description/data/'
name_out_to_in = None
memory = None
def condition_independent_test(input_str):
    try:
        global data, col_name
        print(input_str)
        input_str = json.loads(input_str.split('\n')[0])
        index = input_str['interesting_var']

        condition = None if input_str['condition'] == "None" or input_str['condition'] == [] else input_str['condition']
        postfix = ''
        if condition is not None:
            postfix = ' under conditions : ' + ','.join(condition)
        cit = CIT(data=np.array(data), method='fisherz')
        pValue = cit(data.columns.get_loc(index[0]), data.columns.get_loc(index[1]),
                     [data.columns.get_loc(col) for col in condition] if condition is not None else None)

        if pValue < 0.05:
            return '''{} and {} is not independent'''.format(index[0], index[1])+postfix
        else:
            return '''{} and {} is independent'''.format(index[0], index[1])+postfix
    except Exception as e:
        return f"tool raise a error :{e}. please check your input format and input variables."

def draw_pydot(pyd):
    tmp_png = pyd.create_png(f="png")
    fp = io.BytesIO(tmp_png)
    img = mpimg.imread(fp, format='png')
    plt.rcParams["figure.figsize"] = [20, 12]
    plt.rcParams["figure.autolayout"] = True
    plt.axis('off')
    plt.imshow(img)
    plt.show()


def generate_causalgraph(input_str):
    try:
        global folder, name_out_to_in, memory
        print("\n" + input_str)
        input_str = json.loads(input_str.split('\n')[0])
        filename = input_str['filename']
        method = input_str['method'] if 'method' in input_str.keys() else 'pc'
        interested_var = input_str['interesting_var'] if 'interesting_var' in input_str.keys() else None
        analyse_relationship = input_str['analyse_relationship'] if 'analyse_relationship' in input_str.keys() else None
        config = input_str['config'] if 'config' in input_str.keys() else None
        print(config)
        data_sum = pd.read_csv(os.path.join(folder, filename), header=None)
        data_sum.columns = name_out_to_in
        data = np.array(data_sum)
        if interested_var is not None and interested_var != [] and analyse_relationship == "False":
            data = np.array(data_sum[interested_var])
            cg = pc(data, 0.05, fisherz, node_names=interested_var)
        else:
            cg = pc(data, 0.05, fisherz, node_names=name_out_to_in)
        # cg.draw_pydot_graph()
        name = memory.Get_name(filename)
        memory.add(name, cg)
        with open("tempCG-chatglm-4-plus.txt", "w") as f:
            f.write(str(cg.G))
        return f"causal graph named '{name}' is generate succeed! and have written to the memory, you can use {name} as parameter 'cg_name' "
    except Exception as e:
        return str(e)


def prase_data( config, filename):
    global folder,name_out_to_in
    Y_name = config['Y'] if 'Y' in config.keys() and config['Y'] != [] else None
    Z_name = config['Z'] if 'Z' in config.keys() and config['Z'] != [] else None
    T_name = config['T'] if 'T' in config.keys() and config['T'] != [] else None
    W_name = config['W'] if 'W' in config.keys() and config['W'] != [] else None
    X_name = config['X'] if 'X' in config.keys() and config['X'] != [] else None
    assert Y_name is not None and T_name is not None
    data = pd.read_csv(os.path.join(folder, filename), header=None)
    data.columns = name_out_to_in
    T_mat = data[T_name] if T_name is not None else None
    Y_mat = data[Y_name] if Y_name is not None else None
    Z_mat = data[Z_name] if Z_name is not None else None
    W_mat = data[W_name] if W_name is not None else None
    X_mat = data[X_name] if X_name is not None else None
    return T_mat, Y_mat, Z_mat, W_mat, X_mat,config['T0'],config['T1']


def ATE_sim(input_str):
    try:
        print("\n" + input_str)
        input_str = json.loads(input_str.split('\n')[0])
        filename = input_str['filename']
        config = input_str['config']
        T_mat, Y_mat, Z_mat, W_mat, X_mat,T0,T1 = prase_data(config, filename)
        est = LinearDML(random_state=123)
        est.fit(Y_mat,T_mat,X=X_mat,W=W_mat)
        ate = est.ate(T0=T0,T1=T1,X=X_mat)
        return f'ate : E(Y|T1) - E(Y|T0) is {ate}'
    except Exception as e:
        return f"tools error : {str(e)}"


def empty(input_str):
    print(input_str)
    return "this question doesn't need other tools, so you can only answer the question directly."

def Determine_collider(input_str):
    global folder, name_out_to_in, memory
    try:
        print("\n" + input_str)
        input_str = json.loads(input_str.split('\n')[0])
        method = input_str["method"] if 'method' in input_str.keys() else None
        activate_var = input_str['interesting_var']
        filename = None
        cg = None
        if 'cg_name' in input_str.keys():
            filename = input_str["cg_name"]
            cg, rela = memory.get(filename)
            if rela == Relationship.NO:
                return cg
        else:
            return f"missing cg_name parameter.please input cg_name."
        node_names = [str(i) for i in cg.G.nodes]
        index_a = node_names.index(activate_var[0])
        index_b = node_names.index(activate_var[1])
        rela, varlist = has_collider(index_a, index_b, cg.G.graph)

        if rela == Relationship.YES:
            return "There exists at least one collider {} of {} and {} ".format(node_names[varlist[0]],
                                                                                activate_var[0],
                                                                                activate_var[1])
        if rela == Relationship.UNCERTAIN:
            return "Whether there exist collider of {} and {} is uncertain because some edge direction is uncertain. Following variables may be collider : ".format(
                activate_var[0], activate_var[1]) + ','.join([node_names[i] for i in varlist])

        return "There don't exists collider between {} and {} ".format(activate_var[0], activate_var[1])
    except Exception as e:
        return f"tool raise error : {e}"


def Determine_confounder(input_str):
    try:
        global folder, name_out_to_in, memory
        print("\n" + input_str)
        input_str = json.loads(input_str.split('\n')[0])
        method = input_str["method"] if 'method' in input_str.keys() else None
        activate_var = input_str['interesting_var']
        filename = None
        cg = None
        if 'cg_name' in input_str.keys():
            filename = input_str["cg_name"]
            cg, rela = memory.get(filename)
            if rela == Relationship.NO:
                return cg
        else:
            return f"missing cg_name parameter.please input cg_name."
        node_names = [str(i) for i in cg.G.nodes]
        index_a = node_names.index(activate_var[0])
        index_b = node_names.index(activate_var[1])
        rela, varlist = has_confounder(index_a, index_b, cg.G.graph)

        if rela == Relationship.YES:
            return "yes，there is an unblocked backdoor path between {} and {} so confounder exists. Backdoor path : ".format(
                varlist[0],
                activate_var[0]) + ','.join([node_names[i] for i in varlist])
        if rela == Relationship.UNCERTAIN:
            paths = ''
            for p in varlist:
                paths += "\n" + ','.join([node_names[i] for i in p])
            return "Whether there exist confounder between {} and {} is uncertain because some edge direction is uncertain. Following paths may be backdoor path : ".format(
                activate_var[0], activate_var[1]) + paths
        return "No, no unblocked backdoor paths exist between {} and {}. So don't exist confounder".format(
            activate_var[0], activate_var[1])
    except Exception as e:
        return f"tool raise error : {e}"


def Determine_edge_direction(input_str):
    try:
        global folder, name_out_to_in, memory
        print("\n" + input_str)
        input_str = json.loads(input_str.split('\n')[0])
        filename = None
        cg = None

        if 'cg_name' in input_str.keys():
            filename = input_str["cg_name"]
            cg, rela = memory.get(filename)
            if rela == Relationship.NO:
                return cg
        else:
            return f"missing cg_name parameter.please input cg_name."
        method = input_str["method"] if 'method' in input_str.keys() else None
        activate_var = input_str['interesting_var']
        node_names = [str(i) for i in cg.G.nodes]
        index_a = node_names.index(activate_var[0])
        index_b = node_names.index(activate_var[1])
        a_to_b = cg.G.graph[index_a][index_b]
        b_to_a = cg.G.graph[index_b][index_a]
        prefix = ''
        if a_to_b == -1 and b_to_a == 1:
            return prefix + activate_var[0] + " is a directly cause of " + activate_var[
                1] + ".The opposite is not true"
        elif a_to_b == 1 and b_to_a == -1:
            return prefix + activate_var[1] + " is a cause of " + activate_var[
                0] + ".The opposite is not true"
        elif a_to_b == -1 and b_to_a == -1:
            return prefix + f"An undirected edge exists between  {activate_var[0]} and {activate_var[1]}, which means they are not independent. However, the direction of causality between the two variables is uncertain."
        else:
            return prefix + f"There is no direct edge linking  {activate_var[0]} and {activate_var[1]}.{activate_var[0]} doesn't directly cause {activate_var[1]}."
    except Exception as e:
        return str(e)


from causallearn.search.FCMBased.ANM.ANM import ANM


def Determine_collider(input_str):
    global folder, name_out_to_in, memory
    try:
        print("\n" + input_str)
        input_str = json.loads(input_str.split('\n')[0])
        method = input_str["method"] if 'method' in input_str.keys() else None
        activate_var = input_str['interesting_var']
        filename = None
        cg = None
        if 'cg_name' in input_str.keys():
            filename = input_str["cg_name"]
            cg, rela = memory.get(filename)
            if rela == Relationship.NO:
                return cg
        else:
            return f"missing cg_name parameter.please input cg_name."
        node_names = [str(i) for i in cg.G.nodes]
        index_a = node_names.index(activate_var[0])
        index_b = node_names.index(activate_var[1])
        rela, varlist = has_collider(index_a, index_b, cg.G.graph)

        if rela == Relationship.YES:
            return "There exists at least one collider {} of {} and {} ".format(node_names[varlist[0]],
                                                                                activate_var[0],
                                                                                activate_var[1])
        if rela == Relationship.UNCERTAIN:
            return "Whether there exist collider of {} and {} is uncertain because some edge direction is uncertain. Following variables may be collider : ".format(
                activate_var[0], activate_var[1]) + ','.join([node_names[i] for i in varlist])

        return "There don't exists collider between {} and {} ".format(activate_var[0], activate_var[1])
    except Exception as e:
        return f"tool raise error : {e}"


def Determine_confounder(input_str):
    try:
        global folder, name_out_to_in, memory
        print("\n" + input_str)
        input_str = json.loads(input_str.split('\n')[0])
        method = input_str["method"] if 'method' in input_str.keys() else None
        activate_var = input_str['interesting_var']
        filename = None
        cg = None
        if 'cg_name' in input_str.keys():
            filename = input_str["cg_name"]
            cg, rela = memory.get(filename)
            if rela == Relationship.NO:
                return cg
        else:
            return f"missing cg_name parameter.please input cg_name."
        node_names = [str(i) for i in cg.G.nodes]
        index_a = node_names.index(activate_var[0])
        index_b = node_names.index(activate_var[1])
        rela, varlist = has_confounder(index_a, index_b, cg.G.graph)

        if rela == Relationship.YES:
            return "yes，there is an unblocked backdoor path between {} and {} so confounder exists. Backdoor path : ".format(
                varlist[0],
                activate_var[0]) + ','.join([node_names[i] for i in varlist])
        if rela == Relationship.UNCERTAIN:
            paths = ''
            for p in varlist:
                paths += "\n" + ','.join([node_names[i] for i in p])
            return "Whether there exist confounder between {} and {} is uncertain because some edge direction is uncertain. Following paths may be backdoor path : ".format(
                activate_var[0], activate_var[1]) + paths
        return "No, no unblocked backdoor paths exist between {} and {}. So don't exist confounder".format(
            activate_var[0], activate_var[1])
    except Exception as e:
        return f"tool raise error : {e}"


def Determine_edge_direction(input_str):
    try:
        global folder, name_out_to_in, memory
        print("\n" + input_str)
        input_str = json.loads(input_str.split('\n')[0])
        filename = None
        cg = None

        if 'cg_name' in input_str.keys():
            filename = input_str["cg_name"]
            cg, rela = memory.get(filename)
            if rela == Relationship.NO:
                return cg
        else:
            return f"missing cg_name parameter.please input cg_name."
        method = input_str["method"] if 'method' in input_str.keys() else None
        activate_var = input_str['interesting_var']
        node_names = [str(i) for i in cg.G.nodes]
        index_a = node_names.index(activate_var[0])
        index_b = node_names.index(activate_var[1])
        a_to_b = cg.G.graph[index_a][index_b]
        b_to_a = cg.G.graph[index_b][index_a]
        prefix = ''
        if a_to_b == -1 and b_to_a == 1:
            return prefix + activate_var[0] + " is a directly cause of " + activate_var[
                1] + ".The opposite is not true"
        elif a_to_b == 1 and b_to_a == -1:
            return prefix + activate_var[1] + " is a cause of " + activate_var[
                0] + ".The opposite is not true"
        elif a_to_b == -1 and b_to_a == -1:
            return prefix + f"An undirected edge exists between  {activate_var[0]} and {activate_var[1]}, which means they are not independent. However, the direction of causality between the two variables is uncertain."
        else:
            return prefix + f"There is no direct edge linking  {activate_var[0]} and {activate_var[1]}.{activate_var[0]} doesn't directly cause {activate_var[1]}."
    except Exception as e:
        return str(e)


def myexec(api_plan,memory,data,name_out_to_in):
    print(f"action : {api_plan}")

    api_plan['api_plan_parameter'] = str(api_plan['api_plan_parameter']).replace("\'","\"")
    if api_plan['api_plan'] == 'empty':
        return empty(api_plan['api_plan_parameter'])
    elif api_plan['api_plan'] == 'Determine_collider':
        return Determine_collider(api_plan['api_plan_parameter'])
    elif api_plan['api_plan'] == 'Determine_confounder':
        return Determine_confounder(api_plan['api_plan_parameter'])
    elif api_plan['api_plan'] == 'Determine_edge_directions':
        return Determine_edge_direction(api_plan['api_plan_parameter'])
    elif api_plan['api_plan'] == 'condition independent test':
        return condition_independent_test(api_plan['api_plan_parameter'])
    elif api_plan['api_plan'] == 'Generate Causal':
        return generate_causalgraph(api_plan['api_plan_parameter'])
    elif api_plan['api_plan'] == 'calculate CATE':
        return ATE_sim(api_plan['api_plan_parameter'])
    else :
        return "api error, this api or tools do not exist"

class RestGPT(Chain):
    """Consists of an agent using tools."""

    llm: BaseLLM
    # api_spec: Optional[ReducedOpenAPISpec] = None
    planner: Planner
    api_selector: APISelector
    scenario: str = "tmdb"
    # requests_wrapper: Optional[RequestsWrapper] = None
    simple_parser: bool = False
    return_intermediate_steps: bool = False
    max_iterations: Optional[int] = 5
    max_execution_time: Optional[float] = None
    early_stopping_method: str = "force"

    def __init__(
        self,
        llm: BaseLLM,
        # api_spec: ReducedOpenAPISpec,
        scenario: str,
        # requests_wrapper: RequestsWrapper,
        caller_doc_with_response: bool = False,
        parser_with_example: bool = False,
        simple_parser: bool = False,
        callback_manager: Optional[BaseCallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        # if scenario in ['TMDB', 'Tmdb']:
        #     scenario = 'tmdb'
        # if scenario in ['Spotify']:
        #     scenario = 'spotify' 
        # if scenario not in ['tmdb', 'spotify']:
        #     raise ValueError(f"Invalid scenario {scenario}")
        
        planner = Planner(llm=llm, scenario=scenario)
        api_selector = APISelector(llm=llm, scenario=scenario)

        super().__init__(
            llm=llm, planner=planner, api_selector=api_selector, scenario=scenario,simple_parser=simple_parser, callback_manager=callback_manager, **kwargs
        )

    def save(self, file_path: Union[Path, str]) -> None:
        """Raise error - saving not supported for Agent Executors."""
        raise ValueError(
            "Saving not supported for RestGPT. "
            "If you are trying to save the agent, please use the "
            "`.save_agent(...)`"
        )

    @property
    def _chain_type(self) -> str:
        return "RestGPT"

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return ["query"]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        return self.planner.output_keys
    
    def debug_input(self) -> str:
        print("Debug...")
        return input()

    def _should_continue(self, iterations: int, time_elapsed: float) -> bool:
        if self.max_iterations is not None and iterations >= self.max_iterations:
            return False
        if (
            self.max_execution_time is not None
            and time_elapsed >= self.max_execution_time
        ):
            return False

        return True

    def _return(self, output, intermediate_steps: list) -> Dict[str, Any]:
        self.callback_manager.on_agent_finish(
            output, color="green", verbose=self.verbose
        )
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        return final_output

    def _get_api_selector_background(self, planner_history: List[Tuple[str, str]]) -> str:
        if len(planner_history) == 0:
            return "No background"
        return "\n".join([step[1] for step in planner_history])

    def _should_continue_plan(self, plan) -> bool:
        if re.search("Continue", plan):
            return True
        return False
    
    def _should_end(self, plan) -> bool:
        if re.search("Final Answer", plan):
            return True
        return False

    def _call(
        self,
        inputs,
        memory2,
        data,
        name_out_to_in2,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        query = inputs
        global name_out_to_in,memory
        name_out_to_in = name_out_to_in2
        memory = memory2

        planner_history: List[Tuple[str, str]] = []
        iterations = 0
        time_elapsed = 0.0
        start_time = time.time()
        print(1)
        plan = self.planner.run(input=query, history=planner_history)
        print(f"Planner: {plan}")
        import json
        while self._should_continue(iterations, time_elapsed):
            tmp_planner_history = [plan]
            api_selector_history: List[Tuple[str, str, str]] = []
            api_selector_background = self._get_api_selector_background(planner_history)
            print(2)
            api_plan = self.api_selector.run(plan=plan, background=api_selector_background)
            print(api_plan)
            api_plan = json.loads(api_plan)
            # print(f"before exec :{api_plan}")
            finished = re.match(r"No API call needed.(.*)", api_plan['api_plan'])
            # print(f"finished : {finished}")
            if not finished:
                execution_res = myexec(api_plan,memory,data,name_out_to_in)
                # execution_res = "i know the answer.generate succeed!"
            else:
                execution_res = finished.group(1)
            print("Observation : " + execution_res)
            planner_history.append((plan, execution_res))
            api_selector_history.append((plan, api_plan['api_plan'], execution_res))
            print(3)
            plan = self.planner.run(input=query, history=planner_history)
            print(f"Planner: {plan}")

            # while self._should_continue_plan(plan):
            #     api_selector_background = self._get_api_selector_background(planner_history)
            #     api_plan = json.loads(self.api_selector.run(plan=tmp_planner_history[0], background=api_selector_background, history=api_selector_history, instruction=plan))
                
            #     print(f"before exec :{api_plan}")
            #     finished = re.match(r"No API call needed.(.*)", api_plan['api_plan'])
            #     print(f"finished : {finished}")

            #     if not finished:
            #         execution_res = myexec()
            #     else:
            #         execution_res = finished.group(1)

            #     planner_history.append((plan, execution_res))
            #     api_selector_history.append((plan, api_plan['api_plan'], execution_res))

            #     plan = self.planner.run(input=query, history=planner_history)
            #     print(f"Planner: {plan}")

            if self._should_end(plan):
                break

            iterations += 1
            time_elapsed = time.time() - start_time
            if iterations > 5:
                break

        return {"result": plan}
