from src.cot_utils import *
from src.dataset_utils import *

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())

# Llama-2 model

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name,use_fast=True)
model = LlamaForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"

dataset = get_dataset_df("xstorycloze",'en')
        
generate_response(df = dataset,
                  task = "xstorycloze",
                  task_lang = "English",
                  instr_lang = "English",
                  prompt_setting = "cot",
                  model = model,
                  tokenizer = tokenizer,
                  name = "llama-7b")

# mt_langs = ['Tibetan']

# for lang in mt_langs:

#     dataset = get_translated_dataset_df("xstorycloze",lang)
        
    # generate_response(df = dataset,
    #               task = "xstorycloze",
    #               task_lang = lang,
    #               instr_lang =lang,
    #               prompt_setting = "basic",
    #               model = model,
    #               tokenizer = tokenizer,
    #               name = "llama-7b")
    
    # generate_response(df = dataset,
    #               task = "xstorycloze",
    #               task_lang = lang,
    #               instr_lang ="English",
    #               prompt_setting = "basic",
    #               model = model,
    #               tokenizer = tokenizer,
    #               name = "llama-7b")
    
    # xstorycloze_langs = {'ru' : 'Russian',
#                 'es' : 'Spanish',
#                 'ar' : 'Arabic',
#                 'hi' : 'Hindi',
#                 'id' : 'Indonesian',
#                 'te' : 'Telugu',
#                 'sw' : 'Swahili',
#                 'zh' : 'Chinese',
#                 'my' : 'Burmese',
#                 'eu' : 'Basque'
#                 }

# for lang in xstorycloze_langs:

#     dataset = get_dataset_df("xstorycloze",lang)
        
#     generate_response(df = dataset,
#                   task = "xstorycloze",
#                   task_lang = xstorycloze_langs[lang],
#                   instr_lang = xstorycloze_langs[lang],
#                   prompt_setting = "basic",
#                   model = model,
#                   tokenizer = tokenizer,
#                   name = "llama-7b")
    
#     generate_response(df = dataset,
#                   task = "xstorycloze",
#                   task_lang = xstorycloze_langs[lang],
#                   instr_lang ="English",
#                   prompt_setting = "basic",
#                   model = model,
#                   tokenizer = tokenizer,
#                   name = "llama-7b")