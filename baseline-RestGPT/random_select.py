import json
import random


filename = 'result_5_18_icl_market_xiugai.jsonl'


with open(filename, 'r') as file:
    lines = [line.strip() for line in file.readlines()]

output_filename = 'random_100.jsonl'



selected_lines = random.sample(lines, 100)


selected_records = [json.loads(line) for line in selected_lines]

# 输出或处理选中的记录
with open(output_filename, 'w') as output_file:
    for line in selected_lines:
        output_file.write(line + '\n')

print(f"{output_filename}")