from src.cot_utils import *
from src.dataset_utils import *

# Llama-2 model

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

dataset = get_dataset_df("msvamp",'en')

def generate_response(df, task, task_lang, instr_lang, prompt_setting, model, tokenizer, name):
    pipeline = transformers.pipeline("text-generation",
                                     model=model,
                                     tokenizer=tokenizer,
                                     torch_dtype=torch.float16,
                                     device_map="auto")
    

    responselist = []

    batch_size=1  
    # Batching the DataFrame
    num_batches = len(df) // batch_size + (len(df) % batch_size != 0)
    for i in range(num_batches):
        batch_df = df.iloc[i*batch_size:(i+1)*batch_size]
        for index, row in batch_df.iterrows():
            prompt = get_prompt(row, task, prompt_setting, instr_lang)
            sequences = pipeline(prompt,
                                 do_sample=False,  # Greedy approach
                                 temperature=0.0,  # t=0.0 raises error if do_sample=True
                                 repetition_penalty=1.18,
                                 num_return_sequences=1,
                                 eos_token_id=tokenizer.eos_token_id,
                                 max_new_tokens=500,  # Max tokens
                                 return_full_text=False)

            for seq in sequences:
                print(prompt)
                print(f"Response: {seq['generated_text']}")
                responselist.append(f"Response: {seq['generated_text']}")

    response_df = pd.DataFrame(data=[row.split(sep=", '") for row in responselist])
    response_df.to_csv(f'results/{name}_{task}_{task_lang}_{prompt_setting}_instr_{instr_lang}.csv', sep=';', index=False, header=False)

generate_response(df=dataset,
                              task='msvamp',
                              task_lang="English",        
                              instr_lang="English",       
                              prompt_setting="basic",   
                              model=model,                
                              tokenizer=tokenizer,
                              name="TEST-zonder-batch")