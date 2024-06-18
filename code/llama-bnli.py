from src.cot_utils_copy import *
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

# bnli_langs = {'ar' : 'Arabic',
#                 'bg' : 'Bulgarian',
#                 'zh' : 'Chinese',
#                 'fr' : 'French',
#                 'de' : 'German',
#                 'el' : 'Greek',
#                 'hi' : 'Hindi',
#                 'ru' : 'Russian',
#                 'es' : 'Spanish',
#                 'sw' : 'Swahili',
#                 'th' : 'Thai',
#                 'tr' : 'Turkish',
#                 'ur' : 'Urdu',
#                 'vi' : 'Vietnamese'
#                 }


mt_langs = ["Ukrainian", "Zulu",'Telugu','Tamil','Tibetan' ]


for lang in mt_langs:

    dataset = get_translated_dataset_df("bnli",lang)
        
    generate_response(df = dataset,
                  task = "bnli",
                  task_lang = lang,
                  instr_lang =lang,
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "llama-2-7b")
    
    generate_response(df = dataset,
                  task = "bnli",
                  task_lang = lang,
                  instr_lang ="English",
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "llama-2-7b")