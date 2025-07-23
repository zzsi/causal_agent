import json
import os
import sys
from datetime import datetime
from io import StringIO

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from util import has_confounder, has_collider, Relationship, Dataloader, CGMemory

f = open('result-gpt3.5-baseline/logging/logging_baseline_ate.txt', 'a+', encoding='utf=8')
sys.stdout = f
now = datetime.now()
print(now)

api_key = ''
model = ChatOpenAI(temperature=0.5, openai_api_key=api_key,
                 model_name="gpt-4o", openai_api_base='')
output_name = 'baseline-gpt3.5-ate.jsonl'
data_loader = Dataloader()
data_loader.read_data('./dataset_ate_gt.json')
skip_num = 0
if os.path.exists(output_name):
    with open(output_name,'r') as f:
        skip_num = len(f.readlines())
        print(f'skip {skip_num}')

package_define = """
import pandas as pd
"""

def _sanitize_output(text: str):
    # print(text)
    _, after = text.split("```python")
    print(f"{package_define}"+after.split("```")[0])
    return f"{package_define}"+after.split("```")[0]

icl_demo_cg = """

#DEMO (show how to use 'df' rather than read from csv file):

I want to know the size of filename.csv

#ANSWER:

```python
#remember use pandas package 
import pandas as pd
#data has been loaded in df, we can use directly
print(len(df))
```

if you need generate causal graph, please use pc package of causallearn and write the causal graph as string format in the file 'causal_graph.txt'.
for example:
```python
from causallearn.search.ConstraintBased.PC import pc
import numpy as np

# If you are interested in a causal graph containing a subset of nodes, slice it first such as df = df[{{node_names}}]. Otherwise, ignore this line.
#hint. all csv data has been loaded,please use df directly. 
cg = pc(np.array(df), 0.05, 'fisherz', node_names=list(df.columns))
with open("causal_graph.txt", "w") as f: #please write in .txt file
    f.write(str(cg.G))
"""
icl_demo_ate = """

#DEMO (show how to use 'df' rather than read from csv file):

I want to know the size of filename.csv

#ANSWER:

```python
#remember use pandas package 
import pandas as pd
#csv data has been loaded in df, we can use directly,don't read from csv
print(len(df))
```

Data is stored in directory './DGP_description/data/'.So you can also use as follow
```python
df = read_csv('./DGP_description/data/xxx.csv')
```

if you need calcuate ate, use LinearDML in econml package and set random_state=123
for example:
```python
from econml.dml import LinearDML
import numpy as np
est = LinearDML(random_state=123)
#use df directly, **mustn't read from csv files**. 
... = df[...] 
# Y,T,X should be 2D array, so use df[[{{names}}]] rather than df[{{names}}]
est.fit(Y = df[[{{Outcome_names}}]],T = df[[{{Treatment_names}}]],X = df[[{{Confounder_names}}]])
T0 = ... #use a number rather than list. such as 0.2
T1 = ... #use a number rather than list. such as 0.1
ate = est.ate(T0=T0,T1=T1,X=df[[{{Confounder_names}}]])
print({"ate":ate[0]})
```
"""
def get_code_result(code,g_dict):
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    try:
        exec(code, g_dict)
        sys.stdout = old_stdout
        return mystdout.getvalue()
    except Exception as e:
        sys.stdout = old_stdout
        return str(e)


for index in range(664):
    line = data_loader.read_one(1)
    if index < skip_num:
        continue
    q, gt, name, col, q_type = data_loader.Get_question_answer_pair(line)
    data = pd.read_csv(os.path.join('DGP_description/data', name), header=None)
    data.columns = col
    name_out_to_in = col
    if q_type not in ["ATE"]:
        continue

    if q_type in ['IT', "CIT", "MULTCIT", "CAUSE", 'Has-Collider', 'Has-Confounder']:
        q = q + f"\n {name} has been loaded in pandas dataframe 'df'. You can use it directly. Please output as json format {{'answer' : '...'}},answer should be one of ['yes','no','uncertain'],Do not output content other than JSON "
    elif q_type not in ["ATE"]:
        q = q + "\n Please only output the causal graph's name directly.output must be json format {{'answer' : '...'}},answer should be the name of the causal graph such as xxx.txt. you needn't really analyse relationship between variables."
    else:
        q = q + "\n Please only output the value of ate.output must be json format {{'ate' : '...'}}"

    g_dict = {'df': data}
    template = """
You are the causal agent. Write some python code to solve the user's causal problem.
Return only python code in Markdown format.The """+ name +""" has been loaded in pandas dataframe 'df',you can use 'df' directly by df['{{column_name}}']  e.g.:

```python
# don't need read csv again, use 'df' directly
df['...'] = ...
....
print({{'answer' : ...}})
```
hint. The values of tabular data are continuous rather than discrete. Please choose an appropriate method to process tabular data. Please use the Fisherz method for independence test. And pd.crosstab don't suitable for continuous data. There are a total of 1000 entries in the table data.That means len(df)=1000.
if calculate ate, set random seed is 123.
```

"""
    print(f"\n\n index : {index}  Q: {q}")
    prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])
    chain = prompt | model | StrOutputParser() | _sanitize_output
    resp = chain.invoke({"input": q+icl_demo_ate})
    resp = get_code_result(resp,g_dict)
    line['output'] = resp
    print(f"{resp}  gt: {gt}")
    if q_type in ["CAUSALKG", "PARTIAL_CG"]:
        try:
            file_name = json.loads(resp.replace('\'','\"'))['answer']
            with open(os.path.join('DGP_description/causal_graph', gt), "r") as f:
                with open(file_name, "r") as f2:
                    cg_str = f2.read()
                if f.read() == cg_str:
                    line['match'] = "MATCH"
                    print("MATCH")
                else:
                    line['match'] = "MISMATCH"
                    print("NOT MATCH")
        except Exception as e:
            print(e)
            line['match'] = "MISMATCH"
            print("NOT MATCH")
    with open(f'{output_name}', 'a+') as f:
        try:
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')
        except Exception as e:
            line['output'] = f'json dump error.{e}'
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')