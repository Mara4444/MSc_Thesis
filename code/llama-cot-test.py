from src.cot_utils import *
from src.dataset_utils import *



# Llama-2 model

# model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)


# mgsm = get_dataset_df("mgsm","en")

# generate_response(df=mgsm,
#                   task='mgsm',
#                         task_lang="English",        # source language 
#                         instr_lang="English",       # get instruction prompt in this language
#                         prompt_setting="basic",     # 'basic' or 'cot'
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama-2-13b")           # model name for saving to .csv

# generate_response(df=mgsm,
#                   task='mgsm',
#                         task_lang="English",        # source language 
#                         instr_lang="English",       # get instruction prompt in this language
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama-2-13b")           # model name for saving to .csv

# msvamp = get_dataset_df("msvamp","en")

# generate_response(df=msvamp,
#                   task='msvamp',
#                         task_lang="English",        # source language 
#                         instr_lang="English",
#                         prompt_setting="basic",     # 'basic' or 'cot'
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama-2-13b")              # model name for saving to .csv

# generate_response(df=msvamp,
#                   task='msvamp',
#                         task_lang="English",        # source language 
#                         instr_lang="English",
#                         prompt_setting="cot",     # 'basic' or 'cot'
#                         model=model,                
#                         tokenizer=tokenizer,
#                         name="llama-2-13b")              # model name for saving to .csv

coinflip = get_dataset_df("coinflip","eng_Latn")

# generate_response(df = coinflip,
#                   task = "coinflip",
#                   task_lang = "English",
#                   instr_lang = "English",
#                   prompt_setting = "basic",
#                   model = model,
#                   tokenizer = tokenizer,
#                   name = "llama-2-13b")
    
generate_response(df = coinflip,
                  task = "coinflip",
                  task_lang = "English",
                  instr_lang = "English",
                  prompt_setting = "cot",
                  model = model,
                  tokenizer = tokenizer,
                  name = "llama-2-13b")

shuffled_objects = get_dataset_df("shuffled_objects","eng_Latn")

generate_response(df = shuffled_objects,
                  task = "shuffled_objects",
                  task_lang = "English",
                  instr_lang = "English",
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "llama-2-13b")
    
generate_response(df = shuffled_objects,
                  task = "shuffled_objects",
                  task_lang = "English",
                  instr_lang = "English",
                  prompt_setting = "cot",
                  model = model,
                  tokenizer = tokenizer,
                  name = "llama-2-13b")

xcopa = get_dataset_df("xcopa","en")

generate_response(df = xcopa,
                  task = "xcopa",
                  task_lang = "English",
                  instr_lang = "English",
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "llama-2-13b")

xcopa = get_dataset_df("xcopa","en")

generate_response(df = xcopa,
                  task = "xcopa",
                  task_lang = "English",
                  instr_lang = "English",
                  prompt_setting = "cot",
                  model = model,
                  tokenizer = tokenizer,
                  name = "llama-2-13b")