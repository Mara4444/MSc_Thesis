from src.cot_utils_copy import *
from src.dataset_utils import *

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())

# Bloomz model

model_name = "bigscience/bloomz-7b1-mt"
model = BloomForCausalLM.from_pretrained(model_name)
tokenizer = BloomTokenizerFast.from_pretrained(model_name)

tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"

# xcopa_langs = {'et' : 'Estonian',
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

xcopa_langs = {'qu' : 'Quechua',
                'zh' : 'Chinese'}

for lang in xcopa_langs:
    
    dataset = get_dataset_df("xcopa",lang)
            
    generate_response(df = dataset,
                    task = "xcopa",
                    task_lang = xcopa_langs[lang],
                    instr_lang = xcopa_langs[lang],
                    prompt_setting = "basic",
                    model = model,
                    tokenizer = tokenizer,
                    name = "bloomz-7b1-rerun")

mt_langs = ['Quechua','Chinese']

for lang in mt_langs:
    
    dataset = get_translated_dataset_df("xcopa",lang)
            
    generate_response(df = dataset,
                    task = "xcopa",
                    task_lang = lang,
                    instr_lang = lang,
                    prompt_setting = "basic",
                    model = model,
                    tokenizer = tokenizer,
                    name = "sanity-bloomz-xcopa-rerun")
    

mt_langs = ['Arabic','Belarusian','Bengali','Tibetan',
                                 'Bulgarian', 'Maithili', "Malayalam", 'Marathi', 'Nepali',
                                 'Hebrew','Hindi','Croatian','Russian',"Serbian", "Ukrainian", "Urdu"]

for lang in mt_langs:
    
    dataset = get_translated_dataset_df("xcopa",lang)
            
    generate_response(df = dataset,
                    task = "xcopa",
                    task_lang = lang,
                    instr_lang = lang,
                    prompt_setting = "basic",
                    model = model,
                    tokenizer = tokenizer,
                    name = "bloomz-7b1-rerun")
    

# mt_langs = ['Afrikaans','Arabic','Balinese','Belarusian','Bengali','Tibetan', 'Bosnian', 
#                                  'Bulgarian', 'Catalan', 'Czech', 'Danish', 'Khmer', 'Korean', 'Lao', 'Maithili', 
#                                  'Malayalam', 'Marathi', 'Dutch', 'Norwegian', 'Nepali', 'German', 'Polish', 'Greek',
#                                  'Portuguese','Russian','French','Romanian','Finnish','Hebrew','Slovak','Hindi',
#                                  'Croatian','Hungarian','Swedish','Japanese','Javanese',"Armenian", "Bulgarian", 
#                                  "Burmese", "Cantonese", "Malay", "Serbian", "Slovenian", "Spanish", "Tagalog", 
#                                  "Telugu", "Ukrainian", "Urdu", "Zulu"]




