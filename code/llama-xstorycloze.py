from src.cot_utils_copy import *
from src.dataset_utils import *

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())

# Llama-2 model

# model_name = "meta-llama/Llama-2-7b-chat-hf"
# tokenizer = LlamaTokenizer.from_pretrained(model_name,use_fast=True)
# model = LlamaForCausalLM.from_pretrained(model_name)

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"

xstorycloze_langs = {'en' : 'English'}

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

for lang in xstorycloze_langs:

    dataset = get_dataset_df("xstorycloze",lang)
        
    generate_response(df = dataset,
                  task = "xstorycloze",
                  task_lang = xstorycloze_langs[lang],
                  instr_lang =xstorycloze_langs[lang],
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "llama-3-8b")
    
    # generate_response(df = dataset,
    #               task = "xstorycloze",
    #               task_lang = xstorycloze_langs[lang],
    #               instr_lang ="English",
    #               prompt_setting = "basic",
    #               model = model,
    #               tokenizer = tokenizer,
    #               name = "llama-3-8b")
