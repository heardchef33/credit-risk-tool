import pytest 
import pandas as pd 

@pytest.fixture
def sample_input_data(): 

    data = {
        "int_rate" : [10, 12],
        "fico_range_high" : [600, 550],
        "inq_last_6mths" : [1.0, 2.0],
        "open_il_12m" : [1.0, 2.0],
        "acc_open_past_24mths" : [2.0, 3.0],
        "mort_acc" : [2, 1],
        "num_tl_op_past_12m" : [3.0, 3.0],
        "percent_bc_gt_75" : [4.0, 2.0],
        "sub_grade" : ['A1', 'G1'],
        "term" : ['36 months', '60 months']
    }

    return pd.DataFrame(data)