from src.cot_utils_copy import *
from src.dataset_utils import *

# Llama-2 model

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name,use_fast=True)
model = LlamaForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"

dataset = get_dataset_df("msvamp",'th')

generate_response(df=dataset,
                              task='msvamp',
                              task_lang="Thai",        
                              instr_lang="Thai",       
                              prompt_setting="cot",   
                              model=model,                
                              tokenizer=tokenizer,
                              name="llama-2-7b") 

langs = {'zh' : 'Chinese',
         'ja' : 'Japanese'}

for lang in langs:

      dataset = get_dataset_df("msvamp",lang)

      generate_response(df=dataset,
                              task='msvamp',
                              task_lang=langs[lang],        
                              instr_lang=langs[lang],       
                              prompt_setting="basic",   
                              model=model,                
                              tokenizer=tokenizer,
                              name="llama-2-7b")         

      generate_response(df=dataset,
                              task='msvamp',
                              task_lang=langs[lang],        
                              instr_lang=langs[lang],       
                              prompt_setting="cot",   
                              model=model,                
                              tokenizer=tokenizer,
                              name="llama-2-7b")  

# mt_langs = ['Afrikaans','Arabic','Balinese','Belarusian','Tibetan', 'Bosnian', 'Haitian','Indonesian','Quechua','Bulgarian', 
#             'Catalan', 'Czech', 'Danish', 'Croatian','Hungarian','Swedish', 'Estonian','Khmer', 'Korean', 'Lao', 'Maithili',
#             'Malayalam', 'Marathi', 'Dutch']

# mt_langs = ['Norwegian', 'Nepali', 'Polish', 'Greek', 'Portuguese','Romanian','Finnish','Hebrew','Slovak','Hindi','Italian',
#             'Vietnamese','Javanese',"Armenian",'Turkish', "Burmese", "Cantonese", "Malay", "Serbian", "Slovenian", "Tagalog", "Telugu",
#             'Tamil',"Ukrainian", "Urdu", "Zulu"]
      

# mt_langs = ['Bosnian', 'Haitian','Indonesian','Quechua','Bulgarian', 
#             'Catalan', 'Czech', 'Danish', 'Croatian','Hungarian','Swedish', 'Estonian','Khmer', 'Korean', 'Lao', 'Maithili',
#             'Malayalam', 'Marathi', 'Dutch']
      
# mt_langs = ["Armenian",'Turkish', "Burmese", "Cantonese", "Malay", "Serbian", "Slovenian", "Tagalog", "Telugu",
#             'Tamil',"Ukrainian", "Urdu", "Zulu"]

# for lang in mt_langs:

#       dataset = get_translated_dataset_df("msvamp",lang)

#       generate_response(df=dataset,
#                         task='msvamp',
#                               task_lang=lang,        # source language 
#                               instr_lang=lang,       # get instruction prompt in this language
#                               prompt_setting="basic",     # 'basic' or 'cot'
#                               model=model,                
#                               tokenizer=tokenizer,
#                               name="llama-2-7b")           # model name for saving to .csv