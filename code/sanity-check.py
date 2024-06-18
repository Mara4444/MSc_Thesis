from src.cot_utils_copy import *
from src.dataset_utils import *

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())


model_name = "bigscience/bloomz-7b1-mt"
model = BloomForCausalLM.from_pretrained(model_name)
tokenizer = BloomTokenizerFast.from_pretrained(model_name)
  
mt_langs = ["Arabic", "Bulgarian", "Chinese", "French", "German", "Greek", "Hindi", 
              "Russian", "Spanish", "Swahili", "Thai", "Turkish", "Urdu", "Vietnamese"]
for lang in mt_langs:

    dataset = get_translated_dataset_df("bnli",lang)

    generate_response(df = dataset,
                  task = "bnli",
                  task_lang = lang,
                  instr_lang = lang,
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "sanity-bloomz-bnli")

    generate_response(df = dataset,
                  task = "bnli",
                  task_lang = lang,
                  instr_lang = "English",
                  prompt_setting = "basic",
                  model = model,
                  tokenizer = tokenizer,
                  name = "sanity-bloomz-bnli-eng")


# # Llama-2 model

# model_name = "meta-llama/Llama-2-7b-chat-hf"
# tokenizer = LlamaTokenizer.from_pretrained(model_name)
# model = LlamaForCausalLM.from_pretrained(model_name)
  
# mt_langs = ["Bulgarian", "Chinese", "French", "German",  
#               "Russian", "Spanish", "Swahili", "Turkish","Vietnamese"]

# for lang in mt_langs:

#     dataset = get_translated_dataset_df("bnli",lang)

#     generate_response(df = dataset,
#                   task = "bnli",
#                   task_lang = lang,
#                   instr_lang = lang,
#                   prompt_setting = "basic",
#                   model = model,
#                   tokenizer = tokenizer,
#                   name = "sanity-llama-bnli")

#     generate_response(df = dataset,
#                   task = "bnli",
#                   task_lang = lang,
#                   instr_lang = "English",
#                   prompt_setting = "basic",
#                   model = model,
#                   tokenizer = tokenizer,
#                   name = "sanity-llama-bnli-eng")