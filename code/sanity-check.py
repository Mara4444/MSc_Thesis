from src.cot_utils_copy import *
from src.dataset_utils import *

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())



# Llama-2 model

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"


# mt_langs =["Thai", "Turkish", "Vietnamese", "Chinese"]
    
      
# for lang in mt_langs:

#     dataset = get_translated_dataset_df("xcopa",lang)

#     generate_response(df = dataset,
#                   task = "xcopa",
#                   task_lang = lang,
#                   instr_lang = lang,
#                   prompt_setting = "basic",
#                   model = model,
#                   tokenizer = tokenizer,
#                   name = "sanity-llama-xcopa")
  


# for lang in mt_langs:

#     dataset = get_translated_dataset_df("xcopa",lang)

#     generate_response(df = dataset,
#                   task = "xcopa",
#                   task_lang = lang,
#                   instr_lang = lang,
#                   prompt_setting = "basic",
#                   model = model,
#                   tokenizer = tokenizer,
#                   name = "sanity-llama-xcopa")

#sanity check xstorycloze
mt_langs = ['Swahili','Chinese',"Spanish"]
# mt_langs = ["Telugu", 'Arabic', "Basque","Burmese"]

for lang in mt_langs:

    dataset = get_translated_dataset_df("xstorycloze",lang)

    generate_response(df = dataset,
                  task = "xstorycloze",
                  task_lang = lang,
                  instr_lang = lang,
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "sanity-llama-xstorycloze")
    
# # Bloomz model

# model_name = "bigscience/bloomz-7b1-mt"
# model = BloomForCausalLM.from_pretrained(model_name)
# tokenizer = BloomTokenizerFast.from_pretrained(model_name)
    
#sanity check xstorycloze
# mt_langs = ['Swahili',"Telugu"]

# for lang in mt_langs:

#     dataset = get_translated_dataset_df("xstorycloze",lang)

#     generate_response(df = dataset,
#                   task = "xstorycloze",
#                   task_lang = lang,
#                   instr_lang = lang,
#                   prompt_setting = "basic",
#                   model = model,
#                   tokenizer = tokenizer,
#                   name = "sanity-bloomz-xstorycloze")