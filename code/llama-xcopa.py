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

# xcopa_langs = {'en' : 'English',
#                 'et' : 'Estonian',
#                 'ht' : 'Haitian',
#                 'id' : 'Indonesian',
#                 'qu' : 'Quechua',
#                 'it' : 'Italian',
#                 'sw' : 'Swahili',
#                 'ta' : 'Tamil',
#                 'th' : 'Thai',
#                 'zh' : 'Chinese',
#                 'tr' : 'Turkish',
#                 'vi' : 'Vietnamese'}

# mt_langs = ['Afrikaans','Arabic','Balinese','Belarusian','Bengali','Tibetan', 'Bosnian', 
#                                  'Bulgarian', 'Catalan', 'Czech', 'Danish', 'Khmer', 'Korean', 'Lao', 'Maithili', 
#                                  'Malayalam', 'Marathi', 'Dutch', 'Norwegian', 'Nepali', 'German', 'Polish', 'Greek',
#                                  'Portuguese','Russian','French','Romanian','Finnish','Hebrew','Slovak','Hindi',
#                                  'Croatian','Hungarian','Swedish','Japanese','Javanese',"Armenian", "Bulgarian", 
#                                  "Burmese", "Cantonese", "Malay", "Serbian", "Slovenian", "Spanish", "Tagalog", 
#                                  "Telugu", "Ukrainian", "Urdu", "Zulu","Basque"]

mt_langs = ["Basque"]

for lang in mt_langs:
    dataset = get_translated_dataset_df("xcopa",lang)

    generate_response(df = dataset,
                  task = "xcopa",
                  task_lang = lang,
                  instr_lang = lang,
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "llama-2-7b")
    
    generate_response(df = dataset,
                  task = "xcopa",
                  task_lang = lang,
                  instr_lang = "English",
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "llama-2-7b")