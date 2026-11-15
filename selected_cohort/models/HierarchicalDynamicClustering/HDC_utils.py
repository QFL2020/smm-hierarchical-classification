# Selected-cohort method reported in manuscript Table 1 / Supplement S1–S4.
# See README: outcome-based exclusion and evaluation limitations are intentional here.

def HDCLogWrite(path: str, line: str):
    with open(path + '/model_log.txt', 'a+', buffering=1) as f:
        f.write(line + '\n')

def get_height(node_id: int):
    height = 0
    while node_id != 0:
        height += 1
        node_id //= 2
    return height
