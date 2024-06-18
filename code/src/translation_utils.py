
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from datasets import load_dataset

nltk.download('punkt')

################################################
####            machine translation         ####
################################################

def translate_list(input_list,trg_lang,model,tokenizer):
    """
    Translate a list from English to the target language.
    
    Parameters:
    input_list: input list of strings to translate.
    trg_lang: target language given in nllb-200 code.
    
    Returns:
    Translated list of strings.
    """
    translated_list = []

    for string in input_list:
        
        translated_string = ""
        
        sentences = sent_tokenize(string)

        for sentence in sentences:
            inputs = tokenizer(
                sentence, 
                return_tensors="pt"
                )
        
            translated_tokens = model.generate(
                **inputs, 
                forced_bos_token_id=tokenizer.lang_code_to_id[trg_lang], 
                max_length=100 # set to longer than max length of a sentence in dataset?
                )
            
            translated_sentence = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            print(sentence, translated_sentence)
            translated_string = translated_string + translated_sentence + ' '

        translated_list.append(translated_string)

    return translated_list

def translate_dataset(dataset,name,trg_lang,model,tokenizer): 
    """
    Translate a dataset from English to the target language.
    
    Parameters:
    dataset: dataset to translate.
    name: name of the dataset ['mgsm', 'xcopa', 'xstorycloze', 'mkqa', 'pawsx', 'xnli' or 'xlsum']
    trg_lang: target language given in iso2-code.
    
    Returns:
    Translated dataset and returns as DataFrame. 
    """
    if name  == 'mgsm': 

        translated1_list = translate_list(dataset["question"],trg_lang,model,tokenizer)

        translated_dataset = pd.DataFrame({'question': translated1_list,
                                           'answer_number': dataset["answer_number"]
                                           })

        translated_dataset.to_csv('../datasets/mgsm/mgsm_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset
    
    elif name  == 'xcopa': 

        translated1_list = translate_list(dataset["premise"],trg_lang,model,tokenizer)
        # translated2_list = translate_list(dataset["question"],trg_lang,model,tokenizer)
        translated3_list = translate_list(dataset["choice1"],trg_lang,model,tokenizer)
        translated4_list = translate_list(dataset["choice2"],trg_lang,model,tokenizer)

        translated_dataset = pd.DataFrame({'premise': translated1_list,
                                           'question': dataset["question"],
                                           'choice1': translated3_list,
                                           'choice2': translated4_list, 
                                           'label': dataset["label"]
                                           })

        translated_dataset.to_csv('../datasets/xcopa/xcopa_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset
    
    elif name  == 'msvamp': 

        translated1_list = translate_list(dataset["m_query"],trg_lang,model,tokenizer)

        translated_dataset = pd.DataFrame({'m_query': translated1_list,
                                           'response': dataset["response"]
                                           })

        translated_dataset.to_csv('../datasets/msvamp/msvamp_' + trg_lang + '.csv', sep=';', index=False, header=True)

        return translated_dataset
    
    elif name  == 'xstorycloze': 

        translated1_list = translate_list(dataset["input_sentence_1"],trg_lang,model,tokenizer)
        translated2_list = translate_list(dataset["input_sentence_2"],trg_lang,model,tokenizer)
        translated3_list = translate_list(dataset["input_sentence_3"],trg_lang,model,tokenizer)
        translated4_list = translate_list(dataset["input_sentence_4"],trg_lang,model,tokenizer)
        translated5_list = translate_list(dataset["sentence_quiz1"],trg_lang,model,tokenizer)
        translated6_list = translate_list(dataset["sentence_quiz2"],trg_lang,model,tokenizer)

        translated_dataset = pd.DataFrame({'input_sentence_1': translated1_list,
                                           'input_sentence_2': translated2_list,
                                           'input_sentence_3': translated3_list,
                                           'input_sentence_4': translated4_list, 
                                           'sentence_quiz1': translated5_list,
                                           'sentence_quiz2': translated6_list,
                                           'answer_right_ending': dataset["answer_right_ending"]
                                           })
        translated_dataset.to_csv('../datasets/xstorycloze/xstorycloze_' + trg_lang + '.csv', sep=';', index=False, header=True)
        
        return translated_dataset
    
    elif name  == 'bnli': 

        translated1_list = translate_list(dataset["premise"],trg_lang,model,tokenizer)
        translated2_list = translate_list(dataset["hypothesis"],trg_lang,model,tokenizer)


        translated_dataset = pd.DataFrame({'premise': translated1_list,
                                           'hypothesis': translated2_list,
                                           'label': dataset["label"]
                                           })
        translated_dataset.to_csv('../datasets/bnli/bnli_' + trg_lang + '.csv', sep=';', index=False, header=True)
        
        return translated_dataset

    else:
        print("Dataset name is not correctly specified.")



#######################################################
############# Translate instructions ##################
#######################################################
        

def translate_instruction_cot(languages,model,tokenizer):
    """
    Translate the "Let's think step by step." statement to the target language.
    
    Parameters:
    languages: list of languages available in the nllb model to translate to.
    
    Returns:
    Translated dataset with cot statement and returns as DataFrame. 
    """
    translated_list = []

    for lang in languages:
        
        translated_list.append(translate_string(inputstring="Let's think step by step.",trg_lang=lang,model=model,tokenizer=tokenizer) )

    translated_instructions = pd.DataFrame({'language' : languages,
                                   'cot' : translated_list,
                                   })

    translated_instructions.to_csv('translated_instructions_cot.csv', sep=';', index=False, header=True)
    # translated_instructions.to_csv('translated_instructions_cot_eus_Latn.csv', sep=';', index=False, header=True)

    return translated_instructions

def translate_instruction_xcopa(languages,model,tokenizer):
    """
    Translate the prompt instruction of XCOPA to the target languages.
    
    Parameters:
    languages: list of languages available in the nllb model to translate to.
    
    Returns:
    Translated dataset with XCOPA prompt instruction and returns as DataFrame. 
    """
    translated_list1 = []
    translated_list2 = []

    for lang in languages:

        instruction1 = translate_string(inputstring="Premise:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction2 = translate_string(inputstring="Option A:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction3 = translate_string(inputstring="Option B:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction4 = translate_string(inputstring="Based on the premise, which cause is more likely?",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction5 = translate_string(inputstring="Based on the premise, which effect is more likely?",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction6 = translate_string(inputstring="Pick between options A and B.",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction7 = translate_string(inputstring="Answer:",trg_lang=lang,model=model,tokenizer=tokenizer) 

        instruction_cause = instruction1 + ' {premise} \n' + instruction2 + ' {choice1} \n' + instruction3 + '{choice2} \n' + instruction4 + '\n' + instruction6 + '\n' + instruction7 + ' {cot}'
        instruction_effect = instruction1 + ' {premise} \n' + instruction2 + ' {choice1} \n' + instruction3 + '{choice2} \n' + instruction5 + '\n' + instruction6 + '\n' + instruction7 + ' {cot}'
        
        translated_list1.append(instruction_cause)
        translated_list2.append(instruction_effect)

    translated_instructions = pd.DataFrame({'language' : languages,
                                   'xcopa_cause' : translated_list1,
                                   'xcopa_effect' : translated_list2
                                   })

    translated_instructions.to_csv('translated_instructions_xcopa.csv', sep=';', index=False, header=True)
    # translated_instructions.to_csv('translated_instructions_xcopa_eus_Latn.csv', sep=';', index=False, header=True)

    return translated_instructions

def translate_instruction_mgsm(languages,model,tokenizer):
    """
    Translate the prompt instruction of MGSM to the target languages.
    
    Parameters:
    languages: list of languages available in the nllb model to translate to.
    
    Returns:
    Translated dataset with MGSM prompt instruction and returns as DataFrame. 
    """
    translated_list1 = []
    translated_list2 = []

    for lang in languages:

        instruction1 = translate_string(inputstring="Question:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction2 = translate_string(inputstring="Based on the question, formulate a numeric answer.",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction3 = translate_string(inputstring="Answer:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction4 = translate_string(inputstring="Let's think step by step.",trg_lang=lang,model=model,tokenizer=tokenizer)

        instruction_basic = instruction1 + ' {question} \n' + instruction2 + ' ' + instruction3
        instruction_cot = instruction1 + ' {question} \n' + instruction2 + ' ' + instruction3 + ' ' + instruction4
        
        translated_list1.append(instruction_basic)
        translated_list2.append(instruction_cot)

    translated_instructions = pd.DataFrame({'language' : languages,
                                   'mgsm_basic' : translated_list1,
                                   'mgsm_cot' : translated_list2
                                   })

    translated_instructions.to_csv('translated_instructions_mgsm.csv', sep=';', index=False, header=True)
    # translated_instructions.to_csv('translated_instructions_mgsm_eus_Latn.csv', sep=';', index=False, header=True)

    return translated_instructions

def translate_instruction_xstorycloze(languages,model,tokenizer):
    """
    Translate the prompt instruction of xstorycloze to the target languages.
    
    Parameters:
    languages: list of languages available in the nllb model to translate to.
    
    Returns:
    Translated dataset with xstorycloze prompt instruction and returns as DataFrame. 
    """
    translated_list = []

    for lang in languages:

        instruction1 = translate_string(inputstring="Consider the following story:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction2 = translate_string(inputstring="Option A:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction3 = translate_string(inputstring="Option B:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction4 = translate_string(inputstring="Which ending to the story is most likely?",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction5 = translate_string(inputstring="Pick between options A and B.",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction6 = translate_string(inputstring="Answer:",trg_lang=lang,model=model,tokenizer=tokenizer) 

        instruction = instruction1 + ' \n' + '{input_sentence_1} ' + '{input_sentence_2} ' + '{input_sentence_3} ' + '{input_sentence_4} \n' + instruction2 + '{sentence_quiz1} \n' + instruction3 + '{sentence_quiz2} \n' + instruction4 + '\n' + instruction5 + '\n' + instruction6 + ' {cot}'
        
        translated_list.append(instruction)

    translated_instructions = pd.DataFrame({'language' : languages,
                                   'xstorycloze' : translated_list,
                                   })

    translated_instructions.to_csv('translated_instructions_xstorycloze.csv', sep=';', index=False, header=True)
    # translated_instructions.to_csv('translated_instructions_xstorycloze.csv_eus_Latn', sep=';', index=False, header=True)

    return translated_instructions


def translate_instruction_bnli(languages,model,tokenizer):
    """
    Translate the prompt instruction of B-NLI to the target languages.
    
    Parameters:
    languages: list of languages available in the nllb model to translate to.
    
    Returns:
    Translated dataset with XCOPA prompt instruction and returns as DataFrame. 
    """
    translated_list1 = []
 
    for lang in languages:

        instruction1 = translate_string(inputstring="Premise:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction2 = translate_string(inputstring="Hypothesis:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction3 = translate_string(inputstring="Does the premise entail the hypothesis?",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction4 = translate_string(inputstring="Pick between yes or no.",trg_lang=lang,model=model,tokenizer=tokenizer) 
                
        instruction = instruction1 + ' {premise} \n' + instruction2 + ' {hypothesis} \n' + instruction3 + ' \n' + instruction4 + '\n' 
        
        translated_list1.append(instruction)

    translated_instructions = pd.DataFrame({'language' : languages,
                                   'bnli' : translated_list1
                                   })

    translated_instructions.to_csv('translated_instructions_bnli.csv', sep=';', index=False, header=True)

    return translated_instructions

def translate_instruction_bnli2(languages,model,tokenizer):
    """
    Translate the prompt instruction of B-NLI to the target languages.
    
    Parameters:
    languages: list of languages available in the nllb model to translate to.
    
    Returns:
    Translated dataset with XCOPA prompt instruction and returns as DataFrame. 
    """
    translated_list1 = []
 
    for lang in languages:

        instruction1 = translate_string(inputstring="Premise:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction2 = translate_string(inputstring="Hypothesis:",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction3 = translate_string(inputstring="Does the premise entail the hypothesis?",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction4 = translate_string(inputstring="Option A: Yes.",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction5 = translate_string(inputstring="Option B: No.",trg_lang=lang,model=model,tokenizer=tokenizer)
        instruction6 = translate_string(inputstring="Pick between option A and B.",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction7 = translate_string(inputstring="Answer:",trg_lang=lang,model=model,tokenizer=tokenizer) 

                
        instruction = instruction1 + ' {premise} \n' + instruction2 + ' {hypothesis} \n' + instruction3 + ' \n' + instruction4 + ' \n' + instruction5 + ' \n' + instruction6 + ' \n' + instruction7 
        
        translated_list1.append(instruction)

    translated_instructions = pd.DataFrame({'language' : languages,
                                   'bnli' : translated_list1
                                   })

    translated_instructions.to_csv('translated_instructions_bnli2.csv', sep=';', index=False, header=True)

    return translated_instructions

def translate_instruction_yesno(languages,model,tokenizer):
    """
    Translate yes and no to the target languages.
    
    Parameters:
    languages: list of languages available in the nllb model to translate to.
    
    Returns:
    Translated dataset with XCOPA prompt instruction and returns as DataFrame. 
    """
    translated_list1 = []
    translated_list2 = []
 
    for lang in languages:

        instruction1 = translate_string(inputstring="Yes",trg_lang=lang,model=model,tokenizer=tokenizer) 
        instruction2 = translate_string(inputstring="No",trg_lang=lang,model=model,tokenizer=tokenizer) 
    
        translated_list1.append(instruction1)
        translated_list2.append(instruction2)

    translated_instructions = pd.DataFrame({'language' : languages,
                                   'yes' : translated_list1,
                                   'no' : translated_list2
                                   })

    translated_instructions.to_csv('translated_instructions_yesno.csv', sep=';', index=False, header=True)

    return translated_instructions

        
        
def translate_string(inputstring,trg_lang,model,tokenizer):
    """
    Translate a string from English to the target language.
    
    Parameters:
    inputstringt: input string to translate.
    trg_lang: target language given in nllb-200 code.
    
    Returns:
    Translated string.
    """
    
    translated_string = ""
        
    sentences = sent_tokenize(inputstring)

    for sentence in sentences:
        inputs = tokenizer(
            sentence, 
            return_tensors="pt"
            )
        
        translated_tokens = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.lang_code_to_id[trg_lang], 
            max_length=100
            )
            
        translated_sentence = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        print(sentence, translated_sentence)
        translated_string = translated_string + translated_sentence + ' '

    return translated_string

