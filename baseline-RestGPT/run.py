import os
import json
import logging
import datetime
import time
import yaml

# import spotipy
from langchain.requests import Requests
from langchain import OpenAI

from utils import reduce_openapi_spec, ColorPrint
from model import RestGPT

logger = logging.getLogger()
import tqdm
import pandas as pd
output_name = 'resGPT.jsonl'
import sys
from datetime import datetime


def main():

    os.environ["OPENAI_API_KEY"] =''
    os.environ["OPENAI_API_BASE"] =''

        
    logger.setLevel(logging.INFO)

    # scenario = input("Please select a scenario (TMDB/Spotify): ")
    # scenario = scenario.lower()
    scenario = 'causal'
    # if scenario == 'tmdb':
    # with open("specs/tmdb_oas.json") as f:
    #     raw_tmdb_api_spec = json.load(f)

    # api_spec = reduce_openapi_spec(raw_tmdb_api_spec, only_required=False)

    # # access_token = os.environ["TMDB_ACCESS_TOKEN"]
    # headers = {
    #     'Authorization': f'Bearer {123}'
    # }
    # elif scenario == 'spotify':
    #     with open("specs/spotify_oas.json") as f:
    #         raw_api_spec = json.load(f)

    #     api_spec = reduce_openapi_spec(raw_api_spec, only_required=False, merge_allof=True)

    #     scopes = list(raw_api_spec['components']['securitySchemes']['oauth_2_0']['flows']['authorizationCode']['scopes'].keys())
    #     access_token = spotipy.util.prompt_for_user_token(scope=','.join(scopes))
    #     headers = {
    #         'Authorization': f'Bearer {access_token}'
    #     }
    # else:
    #     raise ValueError(f"Unsupported scenario: {scenario}")
    from langchain_openai import ChatOpenAI
    import warnings

# 忽略所有警告
    warnings.filterwarnings('ignore')
    # requests_wrapper = Requests(headers=headers)
    api_key = ''
    llm = OpenAI(model_name="gpt-4o", temperature=0.0, max_tokens=700)
    rest_gpt = RestGPT(llm,  scenario=scenario, simple_parser=False)

    # if scenario == 'tmdb':
    #     query_example = "Give me the number of movies directed by Sofia Coppola"
    # elif scenario == 'spotify':
    #     query_example = "Add Summertime Sadness by Lana Del Rey in my first playlist"
        
    # print(f"Example instruction: {query_example}")
    # query = input("Please input an instruction (Press ENTER to use the example instruction): ")
    from model.myutil import Dataloader,CGMemory
    data_loader = Dataloader()
    data_loader.read_data('./random_100.jsonl')
    skip_num = 0
    with open(output_name,'r') as f:
        skip_num = len(f.readlines())
    print(f"{skip_num}")
    for index in tqdm.trange(664):
        line = data_loader.read_one(1)
        if index < skip_num:
            continue
        q, gt, name, col, q_type = data_loader.Get_question_answer_pair(line)
        data = pd.read_csv(os.path.join('DGP_description/data', name), header=None)
        data.columns = col
        name_out_to_in = col
        if q_type in ['IT', "CIT", "MULTCIT", "CAUSE", 'Has-Collider', 'Has-Confounder']:
            q = q + "\n Please output as json format {{'answer' : '...'}},answer should be one of ['yes','no','uncertain'],Do not output content other than JSON "
        elif q_type not in ["ATE"]:
            q = q + "\n Please only output the causal graph's name directly.output must be json format {{'answer' : '...'}},answer should be the name of the causal graph.you needn't really analyse relationship between variables."
        else:
            q = q + "\n Please only output the value of ate.output must be json format {{'ate' : '...'}}"

        print(q)
        memory = CGMemory()
        resp = rest_gpt._call(q,memory,data,name_out_to_in)
        line['output'] = resp['result']
        print(f"{resp['result']}  gt: {gt}")
        if q_type in ["CAUSALKG", "PARTIAL_CG"]:
            with open(os.path.join('DGP_description/causal_graph', gt), "r") as f:
                with open("tempCG-chatglm-4-plus.txt", "r") as f2:
                    cg_str = f2.read()
                if f.read() == cg_str:
                    line['match'] = "MATCH"
                    print("MATCH")
                else:
                    line['match'] = "MISMATCH"
                    print("NOT MATCH")
        with open(f'{output_name}', 'a+') as f:
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')
        memory.clear()

if __name__ == '__main__':
    main()
