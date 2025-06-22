import json
file_names = ['gpt3.5-icl-medical.jsonl']
# file_names = ['./baseline_gpt3.5_market_variable_level.jsonl','./baseline_gpt3.5_medical_variable_level.jsonl']
# file_names = ['./baseline-gpt3.5-market-cg.jsonl','./baseline-gpt3.5-medical-cg.jsonl']
# type_names = ["IT","CIT","MULTCIT","CAUSE","Has-Collider","Has-Confounder","CAUSALKG","PARTIAL_CG"]
type_names = ["CAUSE","Has-Collider","Has-Confounder"]
type_names = ["CAUSALKG","PARTIAL_CG"]
type_names = ["IT","CIT","MULTCIT","CAUSE","Has-Collider","Has-Confounder"]
to_read = []
error_items = []
for file_name in file_names:
    with open(file_name,'r') as f:
        for line in f:
            to_read.append(json.loads(line))

print(f"length : {len(to_read)}")

for types in type_names:
    number = 0
    correct_num = 0
    attr_num = {}
    for item in to_read:
        if item['question_type'] == types:
            number += 1
            if item['gt'] not in attr_num.keys():
                attr_num[item['gt']] = [0,0]
            attr_num[item['gt']][0] += 1
            if types in ["IT","CIT","MULTCIT","CAUSE","Has-Collider","Has-Confounder"]:
                try:
                    output = json.loads(item['output'].replace('\'',"\""))
                    if output['answer'] == item['gt']:
                        correct_num += 1
                        attr_num[item['gt']][1] += 1
                    else:
                        error_items.append(item)
                except:
                    error_items.append(item)
                    continue
            if types in ['ATE']:
                try:
                    output = json.loads(item['output'].replace('\'', "\""))
                    if round(float(output['ate']), 5) == round(item['gt'], 5):
                        correct_num += 1
                        attr_num[item['gt']][1] += 1
                    else:
                        error_items.append(item)
                except:
                    error_items.append(item)
                    continue

            if types in ["CAUSALKG","PARTIAL_CG"]:
                try:
                    if item['match'] == "MATCH":
                        correct_num += 1
                        attr_num[item['gt']][1] += 1
                    else:
                        error_items.append(item)
                except:
                    error_items.append(item)
                    continue
    print(f'type {types}, total {number}, correct {correct_num}, rate {correct_num/number}')
    for keys in attr_num.keys():
        print(f"    >>>keys: {keys} ,total {attr_num[keys][0]} ,correct {attr_num[keys][1]}, rate {attr_num[keys][1]/attr_num[keys][0]}")
