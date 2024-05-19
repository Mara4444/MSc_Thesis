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

# xstorycloze_langs = {
    # 'ru' : 'Russian',
    #             'es' : 'Spanish',
    #             'ar' : 'Arabic',
    #             'hi' : 'Hindi',
    #             'id' : 'Indonesian',
    #             'te' : 'Telugu',
    #             'sw' : 'Swahili',
    #             'zh' : 'Chinese',
    #             'my' : 'Burmese',
                # 'eu' : 'Basque'
                # }


# for lang in xstorycloze_langs:

#     dataset = get_dataset_df("xstorycloze",lang)
        
#     generate_response(df = dataset,
#                   task = "xstorycloze",
#                   task_lang = xstorycloze_langs[lang],
#                   instr_lang =xstorycloze_langs[lang],
#                   prompt_setting = "basic",
#                   model = model,
#                   tokenizer = tokenizer,
#                   name = "bloomz-7b1")
    
#     generate_response(df = dataset,
#                   task = "xstorycloze",
#                   task_lang = xstorycloze_langs[lang],
#                   instr_lang ="English",
#                   prompt_setting = "basic",
#                   model = model,
#                   tokenizer = tokenizer,
#                   name = "bloomz-7b1")

# mt_langs = ['Afrikaans','Balinese', 'Belarusian','Bengali','Bosnian', 
#                                  'Bulgarian', 'Catalan', 'Czech', 'Danish', 'Estonian','Haitian','Italian','Khmer', 'Korean', 'Lao', 'Maithili', 
#                                  'Malayalam', 'Marathi', 'Dutch', 'Norwegian', 'Nepali', 'German', 'Polish', 'Greek',
#                                  'Portuguese','Quechua','French','Romanian','Finnish','Hebrew','Slovak',
#                                  'Croatian','Hungarian','Swedish','Japanese','Javanese',"Armenian", "Bulgarian", 
#                                  "Cantonese", "Malay", "Serbian", "Slovenian", "Tagalog", 'Tamil',
#                                  'Thai','Turkish',"Ukrainian", "Urdu",'Vietnamese', "Zulu",'Tibetan']

mt_langs = ['Tibetan']
    
for lang in mt_langs:

    dataset = get_translated_dataset_df("xstorycloze",lang)
        
    generate_response(df = dataset,
                  task = "xstorycloze",
                  task_lang = lang,
                  instr_lang =lang,
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "bloomz-7b1")
    
    generate_response(df = dataset,
                  task = "xstorycloze",
                  task_lang = lang,
                  instr_lang ="English",
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "bloomz-7b1")