from src.cot_utils_copy import *
from src.dataset_utils import *

# Llama-2 model

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name,use_fast=True)
model = LlamaForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"

# mgsm_langs = {'fr' : 'French',
#                 'es' : 'Spanish',
#                 'te' : 'Telugu',
#                 'de' : 'German',
#                 'bn' : 'Bengali',
#                 'sw' : 'Swahili',
#                 'ru' : 'Russian',
#                 'th' : 'Thai',
#                 'zh' : 'Chinese',
#                 'ja' : 'Japanese'}


# mt_langs = ['Afrikaans','Arabic','Balinese','Belarusian','Tibetan', 'Bosnian', 'Haitian','Indonesian','Quechua','Bulgarian', 
#                                  'Catalan', 'Czech', 'Danish', 'Croatian','Hungarian','Swedish', 'Estonian','Khmer', 'Korean', 
#                                  'Lao', 'Maithili','Malayalam', 'Marathi', 'Dutch', 'Norwegian', 'Nepali', 'Polish', 'Greek',
#                                  'Portuguese','Romanian','Finnish','Hebrew','Slovak','Hindi','Italian','Vietnamese',
#                                  'Javanese',"Armenian",'Turkish',"Burmese", "Cantonese", "Malay", 
#                                  "Serbian", "Slovenian", "Tagalog", 'Tamil',"Ukrainian", "Urdu", "Zulu"]

# mt_langs = ['Korean', 'Lao', 'Maithili']
mt_langs = ['Malayalam', "Burmese",'Khmer']

for lang in mt_langs:

      dataset = get_translated_dataset_df("mgsm",lang)

      generate_response(df=dataset,
                        task='mgsm',
                              task_lang=lang,        
                              instr_lang=lang,           
                              prompt_setting="cot",    
                              model=model,                
                              tokenizer=tokenizer,
                              name="llama-2-7b")         