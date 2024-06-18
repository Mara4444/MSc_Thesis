from src.cot_utils_copy import *
from src.dataset_utils import *

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())

# Llama-2 model

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name,use_fast=True)
model = LlamaForCausalLM.from_pretrained(model_name)

# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"

xcopa_langs = {'zh' : 'Chinese',
               'qu' : 'Quechua'}

for lang in xcopa_langs:
    
    dataset = get_dataset_df("xcopa",lang)
            
    generate_response(df = dataset,
                    task = "xcopa",
                    task_lang = xcopa_langs[lang],
                    instr_lang = xcopa_langs[lang],
                    prompt_setting = "basic",
                    model = model,
                    tokenizer = tokenizer,
                    name = "llama-2-7b-rerun")

mt_langs = ['Chinese','Quechua']

for lang in mt_langs:
    
    dataset = get_translated_dataset_df("xcopa",lang)
            
    generate_response(df = dataset,
                    task = "xcopa",
                    task_lang = lang,
                    instr_lang = lang,
                    prompt_setting = "basic",
                    model = model,
                    tokenizer = tokenizer,
                    name = "sanity-llama-xcopa-rerun")
    
mt_langs = ['Cantonese']

for lang in mt_langs:
    
    dataset = get_translated_dataset_df("xcopa",lang)
            
    generate_response(df = dataset,
                    task = "xcopa",
                    task_lang = lang,
                    instr_lang = lang,
                    prompt_setting = "basic",
                    model = model,
                    tokenizer = tokenizer,
                    name = "llama-2-7b-rerun")
    

# mt_langs = ['Arabic','Belarusian','Bengali','Tibetan', 'Bulgarian', 'Maithili']

# mt_langs = ["Malayalam", 'Marathi', 'Nepali','Hebrew','Hindi','Croatian','Russian',"Serbian", "Ukrainian", "Urdu"]

# for lang in mt_langs:
    
#     dataset = get_translated_dataset_df("xcopa",lang)
            
#     generate_response(df = dataset,
#                     task = "xcopa",
#                     task_lang = lang,
#                     instr_lang = lang,
#                     prompt_setting = "basic",
#                     model = model,
#                     tokenizer = tokenizer,
#                     name = "llama-7b-rerun")

# xcopa_langs = {
#                 'et' : 'Estonian',
#                 'ht' : 'Haitian',
#                 'id' : 'Indonesian',
#                 'qu' : 'Quechua',
#                 'it' : 'Italian'}

# xcopa_langs = {'sw' : 'Swahili',
#                 'ta' : 'Tamil',
#                 'th' : 'Thai',
#                 'zh' : 'Chinese',
#                 'tr' : 'Turkish',
#                 'vi' : 'Vietnamese'}

# dataset = get_dataset_df("xcopa","en")

# generate_response(df = dataset,
#                   task = "xcopa",
#                   task_lang = "English",
#                   instr_lang = "English",
#                   prompt_setting = "basic",
#                   model = model,
#                   tokenizer = tokenizer,
#                   name = "llama-3-8b")


# for lang in xcopa_langs:
#     dataset = get_dataset_df("xcopa",lang)

#     generate_response(df = dataset,
#                   task = "xcopa",
#                   task_lang = lang,
#                   instr_lang = lang,
#                   prompt_setting = "basic",
#                   model = model,
#                   tokenizer = tokenizer,
#                   name = "llama-3-8b")
    
#     generate_response(df = dataset,
#                   task = "xcopa",
#                   task_lang = lang,
#                   instr_lang = "English",
#                   prompt_setting = "basic",
#                   model = model,
#                   tokenizer = tokenizer,
#                   name = "llama-3-8b")
    
# mt_langs = ['Afrikaans','Arabic','Balinese','Belarusian','Bengali','Tibetan', 'Bosnian', 
#                                  'Bulgarian', 'Catalan', 'Czech', 'Danish', 'Khmer', 'Korean', 'Lao', 'Maithili', 
#                                  'Malayalam', 'Marathi', 'Dutch', 'Norwegian', 'Nepali', 'German', 'Polish', 'Greek',
#                                  'Portuguese','Russian','French','Romanian','Finnish','Hebrew','Slovak','Hindi',
#                                  'Croatian','Hungarian','Swedish','Japanese','Javanese',"Armenian", "Bulgarian", 
#                                  "Burmese", "Cantonese", "Malay", "Serbian", "Slovenian", "Spanish", "Tagalog", 
#                                  "Telugu", "Ukrainian", "Urdu", "Zulu","Basque"]