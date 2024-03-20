from src.cot_utils import *
from src.dataset_utils import *

# load model

# model_name = "ai-forever/mGPT"
model_name = 'facebook/xglm-7.5B'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


# model_name = "bigscience/bloomz-7b1-mt"
# model = BloomForCausalLM.from_pretrained(model_name)
# tokenizer = BloomTokenizerFast.from_pretrained(model_name)


mgsm = get_dataset_df("mgsm","en")

generate_response(df=mgsm,
                  task='mgsm',
                        task_lang="English",        # source language 
                        instr_lang="English",       # get instruction prompt in this language
                        prompt_setting="basic",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="xglm-7.5b")           # model name for saving to .csv

generate_response(df=mgsm,
                  task='mgsm',
                        task_lang="English",        # source language 
                        instr_lang="English",       # get instruction prompt in this language
                        prompt_setting="cot",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="xglm-7.5b")           # model name for saving to .csv

msvamp = get_dataset_df("msvamp","en")

generate_response(df=msvamp,
                  task='msvamp',
                        task_lang="English",        # source language 
                        instr_lang="English",
                        prompt_setting="basic",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="xglm-7.5b")              # model name for saving to .csv

generate_response(df=msvamp,
                  task='msvamp',
                        task_lang="English",        # source language 
                        instr_lang="English",
                        prompt_setting="cot",     # 'basic' or 'cot'
                        model=model,                
                        tokenizer=tokenizer,
                        name="xglm-7.5b")              # model name for saving to .csv

coinflip = get_dataset_df("coinflip","eng_Latn")

generate_response(df = coinflip,
                  task = "coinflip",
                  task_lang = "English",
                  instr_lang = "English",
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "xglm-7.5b")
    
generate_response(df = coinflip,
                  task = "coinflip",
                  task_lang = "English",
                  instr_lang = "English",
                  prompt_setting = "cot",
                  model = model,
                  tokenizer = tokenizer,
                  name = "xglm-7.5b")

shuffled_objects = get_dataset_df("shuffled_objects","eng_Latn")

generate_response(df = shuffled_objects,
                  task = "shuffled_objects",
                  task_lang = "English",
                  instr_lang = "English",
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "xglm-7.5b")
    
generate_response(df = shuffled_objects,
                  task = "shuffled_objects",
                  task_lang = "English",
                  instr_lang = "English",
                  prompt_setting = "cot",
                  model = model,
                  tokenizer = tokenizer,
                  name = "xglm-7.5b")

xcopa = get_dataset_df("xcopa","en")

generate_response(df = xcopa,
                  task = "xcopa",
                  task_lang = "English",
                  instr_lang = "English",
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "xglm-7.5b")

xcopa = get_dataset_df("xcopa","en")

generate_response(df = xcopa,
                  task = "xcopa",
                  task_lang = "English",
                  instr_lang = "English",
                  prompt_setting = "cot",
                  model = model,
                  tokenizer = tokenizer,
                  name = "xglm-7.5b")