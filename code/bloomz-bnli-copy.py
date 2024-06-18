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


mt_langs = ['Polish',"Tagalog", 
            'Tamil', "Ukrainian", "Zulu",'Tibetan']

for lang in mt_langs:

    dataset = get_translated_dataset_df("bnli",lang)
        
    generate_response(df = dataset,
                  task = "bnli",
                  task_lang = lang,
                  instr_lang =lang,
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "bloomz-7b1")
    
    generate_response(df = dataset,
                  task = "bnli",
                  task_lang = lang,
                  instr_lang ="English",
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "bloomz-7b1")