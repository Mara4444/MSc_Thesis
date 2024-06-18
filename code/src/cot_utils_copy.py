from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, BloomTokenizerFast, BloomForCausalLM
import transformers
import torch
import torch.nn as nn
import pandas as pd
import random
import re
import numpy as np

language_codes = {'Acehnese (Arabic script)' :	'ace_Arab',
    'Acehnese' : 'ace_Latn',
    'Mesopotamian': 'acm_Arab',
    "Ta'izzi-Adeni": 'acq_Arab',
    'Tunisian': 'aeb_Arab',
    'Afrikaans': 'afr_Latn',
    'South Levantine': 'ajp_Arab',
    'Akan': 'aka_Latn',
    'Amharic': 'amh_Ethi',
    'North Levantine': 'apc_Arab',
    'Arabic': 'arb_Arab',
    'Arabic (Romanized)': 'arb_Latn',
    'Najdi': 'ars_Arab',
    'Moroccan': 'ary_Arab',
    'Egyptian': 'arz_Arab',
    'Assamese': 'asm_Beng',
    'Asturian': 'ast_Latn',
    'Awadhi': 'awa_Deva',
    'Aymara': 'ayr_Latn',
    'South Azerbaijani': 'azb_Arab',
    'North Azerbaijani': 'azj_Latn',
    'Bashkir': 'bak_Cyrl',
    'Bambara': 'bam_Latn',
    'Balinese': 'ban_Latn',
    'Belarusian': 'bel_Cyrl',
    'Bemba': 'bem_Latn',
    'Bengali': 'ben_Beng',
    'Bhojpuri': 'bho_Deva',
    'Banjar (Arabic script)': 'bjn_Arab',
    'Banjar': 'bjn_Latn',
    'Tibetan': 'bod_Tibt',
    'Bosnian': 'bos_Latn',
    'Buginese': 'bug_Latn',
    'Bulgarian': 'bul_Cyrl',
    'Catalan': 'cat_Latn',
    'Cebuano': 'ceb_Latn',
    'Czech': 'ces_Latn',
    'Chokwe': 'cjk_Latn',
    'Central Kurdish': 'ckb_Arab',
    'Tatar': 'crh_Latn',
    'Welsh': 'cym_Latn',
    'Danish': 'dan_Latn',
    'German': 'deu_Latn',
    'Dinka': 'dik_Latn',
    'Dyula': 'dyu_Latn',
    'Dzongkha': 'dzo_Tibt',
    'Greek': 'ell_Grek',
    'English': 'eng_Latn',
    'Esperanto': 'epo_Latn',
    'Estonian': 'est_Latn',
    'Basque': 'eus_Latn',
    'Ewe': 'ewe_Latn',
    'Faroese': 'fao_Latn',
    'Fijian': 'fij_Latn',
    'Finnish': 'fin_Latn',
    'Fon': 'fon_Latn',
    'French': 'fra_Latn',
    'Friulian': 'fur_Latn',
    'Fulfulde': 'fuv_Latn',
    'Gaelic': 'gla_Latn',
    'Irish': 'gle_Latn',
    'Galician': 'glg_Latn',
    'Guarani': 'grn_Latn',
    'Gujarati': 'guj_Gujr',
    'Haitian': 'hat_Latn',
    'Hausa': 'hau_Latn',
    'Hebrew': 'heb_Hebr',
    'Hindi': 'hin_Deva',
    'Chhattisgarhi': 'hne_Deva',
    'Croatian': 'hrv_Latn',
    'Hungarian': 'hun_Latn',
    'Armenian': 'hye_Armn',
    'Igbo': 'ibo_Latn',
    'Ilocano': 'ilo_Latn',
    'Indonesian': 'ind_Latn',
    'Icelandic': 'isl_Latn',
    'Italian': 'ita_Latn',
    'Javanese': 'jav_Latn',
    'Japanese': 'jpn_Jpan',
    'Kabyle': 'kab_Latn',
    'Jingpho': 'kac_Latn',
    'Kamba': 'kam_Latn',
    'Kannada': 'kan_Knda',
    'Kashmiri (Arabic script)': 'kas_Arab',
    'Kashmiri': 'kas_Deva',
    'Georgian': 'kat_Geor',
    'Kanuri (Arabic script)': 'knc_Arab',
    'Kanuri': 'knc_Latn',
    'Kazakh': 'kaz_Cyrl',
    'Kabiyè': 'kbp_Latn',
    'Kabuverdianu': 'kea_Latn',
    'Khmer': 'khm_Khmr',
    'Kikuyu': 'kik_Latn',
    'Kinyarwanda': 'kin_Latn',
    'Kyrgyz': 'kir_Cyrl',
    'Kimbundu': 'kmb_Latn',
    'Kurdish': 'kmr_Latn',
    'Kikongo': 'kon_Latn',
    'Korean': 'kor_Hang',
    'Lao': 'lao_Laoo',
    'Ligurian': 'lij_Latn',
    'Limburgish': 'lim_Latn',
    'Lingala': 'lin_Latn',
    'Lithuanian': 'lit_Latn',
    'Lombard': 'lmo_Latn',
    'Latgalian': 'ltg_Latn',
    'Luxembourgish': 'ltz_Latn',
    'Luba-Kasai': 'lua_Latn',
    'Ganda': 'lug_Latn',
    'Luo': 'luo_Latn',
    'Mizo': 'lus_Latn',
    'Latvian': 'lvs_Latn',
    'Magahi': 'mag_Deva',
    'Maithili': 'mai_Deva',
    'Malayalam': 'mal_Mlym',
    'Marathi': 'mar_Deva',
    'Minangkabau (Arabic script)': 'min_Arab',
    'Minangkabau': 'min_Latn',
    'Macedonian': 'mkd_Cyrl',
    'Plateau Malagasy': 'plt_Latn',
    'Maltese': 'mlt_Latn',
    'Meitei': 'mni_Beng',
    'Mongolian': 'khk_Cyrl',
    'Mossi': 'mos_Latn',
    'Maori': 'mri_Latn',
    'Burmese': 'mya_Mymr',
    'Dutch': 'nld_Latn',
    'Norwegian': 'nno_Latn',
    'Norwegian Bokmål': 'nob_Latn',
    'Nepali': 'npi_Deva',
    'Northern Sotho': 'nso_Latn',
    'Nuer': 'nus_Latn',
    'Nyanja': 'nya_Latn',
    'Occitan': 'oci_Latn',
    'Oromo': 'gaz_Latn',
    'Odia': 'ory_Orya',
    'Pangasinan': 'pag_Latn',
    'Panjabi': 'pan_Guru',
    'Papiamento': 'pap_Latn',
    'Persian': 'pes_Arab',
    'Polish': 'pol_Latn',
    'Portuguese': 'por_Latn',
    'Dari': 'prs_Arab',
    'Pashto': 'pbt_Arab',
    'Quechua': 'quy_Latn',
    'Romanian': 'ron_Latn',
    'Rundi': 'run_Latn',
    'Russian': 'rus_Cyrl',
    'Sango': 'sag_Latn',
    'Sanskrit': 'san_Deva',
    'Santali': 'sat_Olck',
    'Sicilian': 'scn_Latn',
    'Shan': 'shn_Mymr',
    'Sinhala': 'sin_Sinh',
    'Slovak': 'slk_Latn',
    'Slovenian': 'slv_Latn',
    'Samoan': 'smo_Latn',
    'Shona': 'sna_Latn',
    'Sindhi': 'snd_Arab',
    'somali': 'som_Latn',
    'Sotho': 'sot_Latn',
    'Spanish': 'spa_Latn',
    'Tosk Albanian': 'als_Latn',
    'Sardinian': 'srd_Latn',
    'Serbian': 'srp_Cyrl',
    'Swati': 'ssw_Latn',
    'Sundanese': 'sun_Latn',
    'Swedish': 'swe_Latn',
    'Swahili': 'swh_Latn',
    'Silesian': 'szl_Latn',
    'Tamil': 'tam_Taml',
    'Tatar': 'tat_Cyrl',
    'Telugu': 'tel_Telu',
    'Tajik': 'tgk_Cyrl',
    'Tagalog': 'tgl_Latn',
    'Thai': 'tha_Thai',
    'Tigrinya': 'tir_Ethi',
    'Tamasheq': 'taq_Latn',
    'Tamasheq (Tifinagh script)': 'taq_Tfng',
    'Tok Pisin': 'tpi_Latn',
    'Tswana': 'tsn_Latn',
    'Tsonga': 'tso_Latn',
    'Turkmen': 'tuk_Latn',
    'Tumbuka': 'tum_Latn',
    'Turkish': 'tur_Latn',
    'Twi': 'twi_Latn',
    'Tamazight': 'tzm_Tfng',
    'Uyghur': 'uig_Arab',
    'Ukrainian': 'ukr_Cyrl',
    'Umbundu': 'umb_Latn',
    'Urdu': 'urd_Arab',
    'Northern Uzbek': 'uzn_Latn',
    'Venetian': 'vec_Latn',
    'Vietnamese': 'vie_Latn',
    'Waray': 'war_Latn',
    'Wolof': 'wol_Latn',
    'Xhosa': 'xho_Latn',
    'Yiddish': 'ydd_Hebr',
    'Yoruba': 'yor_Latn',
    'Cantonese': 'yue_Hant',
    'Chinese (Simplified)': 'zho_Hans',
    'Chinese': 'zho_Hant',
    'Malay': 'zsm_Latn',
    'Zulu': 'zul_Latn'}

language_codes_inv = {v: k for k, v in language_codes.items()}

def get_prompt(row,task,prompt_setting,instr_lang):
    """
    Generate a string prompt for a given promptsetting,task and instruction language.
    
    Parameters:
    row: dataframe row containing the prompt input.
    prompt_setting: different prompting techniques: 'basic', 'cot'
    task: dataset.
    instr_lang: language of the instruction.

    Returns:    
    String prompt.
    """

    instr_lang = language_codes[instr_lang]

    def generate_message(string,**kwargs):
        return string.format(**kwargs)
    
    instructions = pd.read_csv("./datasets/translated_instructions.csv",sep=';')
    instructions.set_index(instructions['language'],inplace=True)
    instructions = instructions.drop('Unnamed: 0',axis=1)
    instructions = instructions.drop('language',axis=1)

    if task == 'xcopa':

        if prompt_setting == 'cot':

            cot = instructions.loc[instr_lang]['cot']

        elif prompt_setting == 'basic':

            cot = ''

        question = row['question']
        
        if question == 'cause':

            return generate_message(instructions.loc[instr_lang]['xcopa_cause'],
                            premise = row['premise'],
                            choice1 = row['choice1'],
                            choice2 = row['choice2'],
                            question = row['question'],
                            cot=cot)
            
        elif question == 'effect':

            return generate_message(instructions.loc[instr_lang]['xcopa_effect'],
                            premise = row['premise'],
                            choice1 = row['choice1'],
                            choice2 = row['choice2'],
                            question = row['question'],
                            cot=cot)

    elif task == 'mgsm':

        if prompt_setting == 'cot':
        
            return generate_message(instructions.loc[instr_lang]['mgsm_cot'],
                                question = row['question'])
        
        elif prompt_setting == 'basic':
            
            return generate_message(instructions.loc[instr_lang]['mgsm_basic'],
                                question = row['question'])  
        
    elif task == 'msvamp':

        if prompt_setting == 'cot':

            return generate_message(instructions.loc[instr_lang]['mgsm_cot'],
                                question = row['m_query'])
        
        elif prompt_setting == 'basic':

            return generate_message(instructions.loc[instr_lang]['mgsm_basic'],
                                question = row['m_query'])   
        
    elif task == 'xstorycloze':

        if prompt_setting == 'cot':

            cot = instructions.loc[instr_lang]['cot']

            return generate_message(instructions.loc[instr_lang]['xstorycloze'],
                                    input_sentence_1 = row['input_sentence_1'], 
                                    input_sentence_2 = row['input_sentence_2'],
                                    input_sentence_3 = row['input_sentence_3'],
                                    input_sentence_4 = row['input_sentence_4'],
                                    sentence_quiz1 = row['sentence_quiz1'],
                                    sentence_quiz2 = row['sentence_quiz2'],
                                    cot = cot)

        elif prompt_setting == 'basic':

            cot = ''
            
            return generate_message(instructions.loc[instr_lang]['xstorycloze'],
                                    input_sentence_1 = row['input_sentence_1'], 
                                    input_sentence_2 = row['input_sentence_2'],
                                    input_sentence_3 = row['input_sentence_3'],
                                    input_sentence_4 = row['input_sentence_4'],
                                    sentence_quiz1 = row['sentence_quiz1'],
                                    sentence_quiz2 = row['sentence_quiz2'],
                                    cot = cot)
    elif task == 'bnli':

        return generate_message(instructions.loc[instr_lang]['bnli2'],
                                    premise = row['premise'], 
                                    hypothesis = row['hypothesis'])
        
def generate_response(df,task,task_lang,instr_lang,prompt_setting,model,tokenizer,name):
    """
    Generate a text response by a given LLM for prompts in a df.
    
    Parameters:
    df: dataframe with questions and answers of the mgsm benchmark.
    task: name of the task.
    task_lang: the language of the prompts in the dataset.
    instr_lang: the required language for the instruction prompt.
    prompt_setting: different prompting techniques: 'basic', 'cot'. 
    model: initialized model.
    tokenizer: initializer tokenizer.
    name: model name to set in the csv filename.
    
    Returns:
    Text generated respons by the LLM for each prompt in the list.
    """

    promptlist = []
    responselist = []
    
    if task == 'bnli':

        df = df[df['label'] != 1]

    for index, row in df.iterrows():
        promptlist.append(get_prompt(row,task,prompt_setting,instr_lang))
    
    # print(promptlist)

    batch_size = 16

    for i in range(0, len(promptlist), batch_size):
        batch_prompts = promptlist[i:i + batch_size]
        print(i,i+batch_size)
        encodings = tokenizer(batch_prompts, padding=True, return_tensors='pt')

        with torch.no_grad():
            generated_ids = model.generate(**encodings,
                                        do_sample=False,  # greedy approach
                                        temperature=0.0,  # t=0.0 would raise error if do_sample=True
                                        repetition_penalty=1.18,  # penalize the model for repeating itself
                                        num_return_sequences=1,
                                        eos_token_id=tokenizer.eos_token_id,
                                        max_new_tokens=20)

        batch_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        print(batch_responses)
        responselist.extend(batch_responses)

    # print(responselist)

    response = pd.DataFrame(data=[row.split(sep=", '") for row in responselist]) 

    def split_at_last_colon(text):
        """
        Splits the text at the last occurrence of a colon ':'.

        Args:
        text (str): The string to be split.

        Returns:
        str: The part of the string after the last colon.
        """
        before, separator, after = text.rpartition(':')
        return after  
    
    response[0] = response[0].apply(split_at_last_colon) 

    if task == 'xcopa':

        
        title = name + '_xcopa_' + task_lang + '_' + prompt_setting + '_instr_' + instr_lang + '.csv'

    elif task == 'mgsm':

        title = name + '_mgsm_' + task_lang + '_' + prompt_setting + '_instr_' + instr_lang + '.csv'

    elif task == 'msvamp':

        title = name + '_msvamp_' + task_lang + '_' + prompt_setting + '_instr_' + instr_lang + '.csv'

    elif task == 'xstorycloze':

        title = name + '_xstorycloze_' + task_lang + '_' + prompt_setting + '_instr_' + instr_lang + '.csv'

    elif task == 'bnli':

        title = name + '_bnli_' + task_lang + '_' + prompt_setting + '_instr_' + instr_lang + '.csv'

    response.to_csv('results/' + title, sep=';', index=False, header=False)


def calculate_accuracy(df1,df2,task):
    """
    Calculate the accuracy (% correct answers) from two input dfs.
    
    Parameters:
    df1: orginial task English file with correct answer column.
    df2: response task file with predicted answer column.
    task: task name.

    Returns:
    Accuracy score (% of correct answers).
    """

    predicted_answerlist = df2['answer'].tolist()

    if task == 'mgsm' or task == 'msvamp':

        df1.iloc[:,1] = pd.to_numeric(df1.iloc[:,1])
    
        correct_answerlist = df1.iloc[:,1].tolist()

        nr_correct = sum(1 for x, y in zip(correct_answerlist, predicted_answerlist) if abs(x - y) < 1e-3)
        accuracy = round(100*(nr_correct / len(correct_answerlist)),1)

        return accuracy


    elif task == 'xcopa':
        
        correct_answerlist = df1['label'].tolist()

        map_label = {0: 'A', 1: 'B'}

        nr_correct = sum(1 for x, y in zip(correct_answerlist, predicted_answerlist) if map_label[x] == y)
        accuracy = round(100*(nr_correct / len(correct_answerlist)),1)

        return accuracy
    
    elif task == 'xstorycloze':
        
        correct_answerlist = df1['answer_right_ending'].tolist()

        map_label = {1: 'A', 2: 'B'}

        nr_correct = sum(1 for x, y in zip(correct_answerlist, predicted_answerlist) if map_label[x] == y)
        accuracy = round(100*(nr_correct / len(correct_answerlist)),1)

        return accuracy
    
    elif task == 'bnli':
        
        correct_answerlist = df1['label'].tolist()

        map_label = {0: 'yes', 1: 'no', 2: 'no'}

        nr_correct = sum(1 for x, y in zip(correct_answerlist, predicted_answerlist) if map_label[x] == y)
        accuracy = round(100*(nr_correct / len(correct_answerlist)),1)

        return accuracy

    
# def extract_numeric_answer(inputstring):
#     """
#     Finds the numeric answer in the model's response.
    
#     Parameters:
#     inputstring: The model's response.

#     Returns:
#     String value of the last mentioned number.
#     """
#     # Regular expression to find 'the answer is ' followed by a number
#     match = re.search(r'The answer is (\b\d+(?:[,.]\d+)?\b)', inputstring,re.IGNORECASE)

#     if match:
#         # Extract the number after 'the answer is'
#         number = match.group(1)
#         number = number.replace(',', '') # 
#         return pd.to_numeric(number, errors='coerce')
    
#     else:
#         numberlist = re.findall(r'\b\d+(?:[,.]\d+)?\b',inputstring)
        
#         if len(numberlist) > 0:
#             number = numberlist[-1]
#             if number is not None:
#                 number = number.replace(',', '') # 
#                 return pd.to_numeric(number, errors='coerce')
#         else:
#             return 0.0
        
def extract_numeric_answer(inputstring):
    """
    Finds the numeric answer in the model's response.
    
    Parameters:
    inputstring: The model's response.

    Returns:
    String value of the last mentioned number.
    """
    if not isinstance(inputstring, str):
        return 0.0
    
    else:
        
        # Regular expression to find 'the answer is ' followed by a number
        match = re.search(r'The answer is (\b\d+(?:[,.]\d+)?\b)', inputstring,re.IGNORECASE)

        if match:
            # Extract the number after 'the answer is'
            number = match.group(1)
            number = number.replace(',', '') # 
            return pd.to_numeric(number, errors='coerce')
        
        else:
            numberlist = re.findall(r'\b\d+(?:[,.]\d+)?\b',inputstring)
            
            if len(numberlist) > 0:
                number = numberlist[-1]
                if number is not None:
                    number = number.replace(',', '') # 
                    return pd.to_numeric(number, errors='coerce')
            else:
                return 0.0
    
def extract_ab_answer(inputstring):
    """
    Finds the multiple choice answer (A, B or C) in the model's response.
    
    Parameters:
    inputstring: The model's response.

    Returns:
    String value of the multiple choice answer.
    """
    if isinstance(inputstring, str):

        matches = re.findall(r'\b[A|B]\b', inputstring)
        
        if len(matches) != 0:
            return matches[0]
        else: 
            return ''
    else:
        return ''
    
def get_results(df,task,response_loc):
    """
    Reads the response csv and calculates the accuracy for a model on a task.
    
    Parameters:
    df: orginial task English file with correct answer column.
    task: task name
    response_loc: string location of the response csv file.

    Returns:
    Accuracy score (%)
    """
    response = pd.read_csv(response_loc,sep=';',header=None)
    response.rename(columns={0:'response'},inplace=True)
    response = response.map(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)

    answer_list = []

    if task == 'xcopa' or task == 'xstorycloze':

        for i in range(len(response)):
            answer = extract_ab_answer(response.iloc[i,0])
            answer_list.append(answer)

        response['answer'] = answer_list

        return calculate_accuracy(df,response,task)
    
    elif task == 'mgsm' or task == 'msvamp':

        for i in range(len(response)):
            answer = extract_numeric_answer(response.iloc[i,0])
            answer_list.append(answer)

        response['answer'] = answer_list


def extract_yesno_answer(inputstring,lang):
    """
    Finds the multiple choice answer (yes or no) in the model's response.
    
    Parameters:
    inputstring: The model's response.

    Returns:
    String value of the multiple choice answer.
    """
    if isinstance(inputstring, str):
        word_map = {
            'Afrikaans': ['ja', 'nee', 'yes', 'no'],
            'Arabic': ['نعم', 'لا', 'yes', 'no'],
            'Armenian': ['այո', 'ոչ', 'yes', 'no'],
            'Balinese': ['ya', 'tidak', 'yes', 'no'],
            'Basque': ['bai', 'ez', 'yes', 'no'],
            'Belarusian': ['так', 'не', 'yes', 'no'],
            'Bengali': ['হ্যাঁ', 'না', 'yes', 'no'],
            'Bosnian': ['da', 'ne', 'yes', 'no'],
            'Bulgarian': ['да', 'не', 'yes', 'no'],
            'Burmese': ['ဟုတ်ကဲ့', 'မဟုတ်', 'yes', 'no'],
            'Cantonese': ['係', '唔係', 'yes', 'no'],
            'Catalan': ['sí', 'no', 'yes'],
            'Chinese': ['是', '不是', 'yes', 'no'],
            'Croatian': ['da', 'ne', 'yes', 'no'],
            'Czech': ['ano', 'ne', 'yes', 'no'],
            'Danish': ['ja', 'nej', 'yes', 'no'],
            'Dutch': ['ja', 'nee', 'yes', 'no'],
            'English': ['yes', 'no'],
            'Estonian': ['jah', 'ei', 'yes', 'no'],
            'Finnish': ['kyllä', 'ei', 'yes', 'no'],
            'French': ['oui', 'non', 'yes', 'no'],
            'German': ['ja', 'nein', 'yes', 'no'],
            'Greek': ['ναι', 'όχι', 'yes', 'no'],
            'Haitian': ['wi', 'non', 'yes', 'no'],
            'Hebrew': ['כן', 'לא', 'yes', 'no'],
            'Hindi': ['हाँ', 'नहीं', 'yes', 'no'],
            'Hungarian': ['igen', 'nem', 'yes', 'no'],
            'Indonesian': ['ya', 'tidak', 'yes', 'no'],
            'Italian': ['sì', 'no', 'yes'],
            'Javanese': ['ya', 'ora', 'yes', 'no'],
            'Japanese': ['はい', 'いいえ', 'yes', 'no'],
            'Khmer': ['បាទ', 'ទេ', 'yes', 'no'],
            'Korean': ['네', '아니요', 'yes', 'no'],
            'Lao': ['ແມ່ນ', 'ບໍ່', 'yes', 'no'],
            'Maithili': ['हाँ', 'न', 'yes', 'no'],
            'Malay': ['ya', 'tidak', 'yes', 'no'],
            'Malayalam': ['അതെ', 'അല്ല', 'yes', 'no'],
            'Marathi': ['होय', 'नाही', 'yes', 'no'],
            'Nepali': ['हो', 'होइन', 'yes', 'no'],
            'Norwegian': ['ja', 'nei', 'yes', 'no'],
            'Polish': ['tak', 'nie', 'yes', 'no'],
            'Portuguese': ['sim', 'não', 'yes', 'no'],
            'Quechuan': ['arí', 'mana', 'yes', 'no'],
            'Romanian': ['da', 'nu', 'yes', 'no'],
            'Russian': ['да', 'нет', 'yes', 'no'],
            'Serbian': ['да', 'не', 'yes', 'no'],
            'Slovak': ['áno', 'nie', 'yes', 'no'],
            'Slovenian': ['ja', 'ne', 'yes', 'no'],
            'Spanish': ['sí', 'no', 'yes'],
            'Swahili': ['ndiyo', 'hapana', 'yes', 'no'],
            'Swedish': ['ja', 'nej', 'yes', 'no'],
            'Tagalog': ['oo', 'hindi', 'yes', 'no'],
            'Tamil': ['ஆம்', 'இல்லை', 'yes', 'no'],
            'Telugu': ['అవును', 'కాదు', 'yes', 'no'],
            'Thai': ['ใช่', 'ไม่', 'yes', 'no'],
            'Tibetan': ['ཨེ', 'མིན', 'yes', 'no'],
            'Turkish': ['evet', 'hayır', 'yes', 'no'],
            'Ukrainian': ['так', 'ні', 'yes', 'no'],
            'Urdu': ['ہاں', 'نہیں', 'yes', 'no'],
            'Vietnamese': ['có', 'không', 'yes', 'no'],
            'Zulu': ['yebo', 'cha', 'yes', 'no']
            }

        words = word_map.get(lang, ['yes', 'no'])  # Default to English if language not found
        pattern = r'\b(?:' + '|'.join(map(re.escape, words)) + r')\b'

        matches = re.findall(pattern, inputstring, re.IGNORECASE)
        
        if len(matches) != 0:
            return matches[0]
        else: 
            return ''
    else:
        return ''
    
def extract_yesno_answer(inputstring,lang):
    """
    Finds the multiple choice answer (yes or no) in the model's response.
    
    Parameters:
    inputstring: The model's response.

    Returns:
    String value of the multiple choice answer.
    """
    if isinstance(inputstring, str):
        word_map = {
            'Afrikaans': ['ja', 'nee', 'yes', 'no'],
            'Arabic': ['نعم', 'لا', 'yes', 'no'],
            'Armenian': ['այո', 'ոչ', 'yes', 'no'],
            'Balinese': ['ya', 'tidak', 'yes', 'no'],
            'Basque': ['bai', 'ez', 'yes', 'no'],
            'Belarusian': ['так', 'не', 'yes', 'no'],
            'Bengali': ['হ্যাঁ', 'না', 'yes', 'no'],
            'Bosnian': ['da', 'ne', 'yes', 'no'],
            'Bulgarian': ['да', 'не', 'yes', 'no'],
            'Burmese': ['ဟုတ်ကဲ့', 'မဟုတ်', 'yes', 'no'],
            'Cantonese': ['係', '唔係', 'yes', 'no'],
            'Catalan': ['sí', 'no', 'yes'],
            'Chinese': ['是', '不', 'yes', 'no'],
            'Croatian': ['da', 'ne', 'yes', 'no'],
            'Czech': ['ano', 'ne', 'yes', 'no'],
            'Danish': ['ja', 'nej', 'yes', 'no'],
            'Dutch': ['ja', 'nee', 'yes', 'no'],
            'English': ['yes', 'no'],
            'Estonian': ['jah', 'ei', 'yes', 'no'],
            'Finnish': ['kyllä', 'ei', 'yes', 'no'],
            'French': ['oui', 'non', 'yes', 'no'],
            'German': ['ja', 'nein', 'yes', 'no'],
            'Greek': ['ναι', 'όχι', 'yes', 'no'],
            'Haitian': ['wi', 'non', 'yes', 'no'],
            'Hebrew': ['כן', 'לא', 'yes', 'no'],
            'Hindi': ['हाँ', 'नहीं', 'yes', 'no'],
            'Hungarian': ['igen', 'nem', 'yes', 'no'],
            'Indonesian': ['ya', 'tidak', 'yes', 'no'],
            'Italian': ['sì', 'no', 'yes'],
            'Javanese': ['ya', 'ora', 'yes', 'no'],
            'Japanese': ['はい', 'いいえ', 'yes', 'no'],
            'Khmer': ['បាទ', 'ទេ', 'yes', 'no'],
            'Korean': ['네', '아니요', 'yes', 'no'],
            'Lao': ['ແມ່ນ', 'ບໍ່', 'yes', 'no'],
            'Maithili': ['हाँ', 'न', 'yes', 'no'],
            'Malay': ['ya', 'tidak', 'yes', 'no'],
            'Malayalam': ['അതെ', 'അല്ല', 'yes', 'no'],
            'Marathi': ['होय', 'नाही', 'yes', 'no'],
            'Nepali': ['हो', 'होइन', 'yes', 'no'],
            'Norwegian': ['ja', 'nei', 'yes', 'no'],
            'Polish': ['tak', 'nie', 'yes', 'no'],
            'Portuguese': ['sim', 'não', 'yes', 'no'],
            'Quechuan': ['arí', 'mana', 'yes', 'no'],
            'Romanian': ['da', 'nu', 'yes', 'no'],
            'Russian': ['да', 'нет', 'yes', 'no'],
            'Serbian': ['да', 'не', 'yes', 'no'],
            'Slovak': ['áno', 'nie', 'yes', 'no'],
            'Slovenian': ['ja', 'ne', 'yes', 'no'],
            'Spanish': ['sí', 'no', 'yes'],
            'Swahili': ['ndiyo', 'hapana', 'yes', 'no'],
            'Swedish': ['ja', 'nej', 'yes', 'no'],
            'Tagalog': ['oo', 'hindi', 'yes', 'no'],
            'Tamil': ['ஆம்', 'இல்லை', 'yes', 'no'],
            'Telugu': ['అవును', 'కాదు', 'yes', 'no'],
            'Thai': ['ใช่', 'ไม่', 'yes', 'no'],
            'Tibetan': ['ཨེ', 'མིན', 'yes', 'no'],
            'Turkish': ['evet', 'hayır', 'yes', 'no'],
            'Ukrainian': ['так', 'ні', 'yes', 'no'],
            'Urdu': ['ہاں', 'نہیں', 'yes', 'no'],
            'Vietnamese': ['có', 'không', 'yes', 'no'],
            'Zulu': ['yebo', 'cha', 'yes', 'no']
            }

        words = word_map.get(lang, ['yes', 'no'])  # Default to English if language not found
        pattern = r'(?:' + '|'.join(map(re.escape, words)) + r')'

        matches = re.findall(pattern, inputstring, re.IGNORECASE)
        
        if len(matches) != 0:
            return matches[0]
        else: 
            return ''
    else:
        return ''
    
def calculate_accuracy_bnli(df1,df2,task,lang):
    """
    Calculate the accuracy (% correct answers) from two input dfs.
    
    Parameters:
    df1: orginial task English file with correct answer column.
    df2: response task file with predicted answer column.
    task: task name.

    Returns:
    Accuracy score (% of correct answers).
    """

    predicted_answerlist = df2['answer'].tolist()
    # print(predicted_answerlist)

    correct_answerlist = df1['label'].tolist()
    # print(correct_answerlist)

    label_map = {
    'Afrikaans': {'ja': 0, 'nee': 2, 'yes': 0, 'no': 2},
    'Arabic': {'نعم': 0, 'لا': 2, 'yes': 0, 'no': 2},
    'Armenian': {'այո': 0, 'ոչ': 2, 'yes': 0, 'no': 2},
    'Balinese': {'ya': 0, 'tidak': 2, 'yes': 0, 'no': 2},
    'Basque': {'bai': 0, 'ez': 2, 'yes': 0, 'no': 2},
    'Belarusian': {'так': 0, 'не': 2, 'yes': 0, 'no': 2},
    'Bengali': {'হ্যাঁ': 0, 'না': 2, 'yes': 0, 'no': 2},
    'Bosnian': {'da': 0, 'ne': 2, 'yes': 0, 'no': 2},
    'Bulgarian': {'да': 0, 'не': 2, 'yes': 0, 'no': 2},
    'Burmese': {'ဟုတ်ကဲ့': 0, 'မဟုတ်': 2, 'yes': 0, 'no': 2},
    'Cantonese': {'係': 0, '唔係': 2, 'yes': 0, 'no': 2},
    'Catalan': {'sí': 0, 'no': 2, 'yes': 0, 'no': 2},
    'Chinese': {'是': 0, '不': 2, 'yes': 0, 'no': 2},
    'Croatian': {'da': 0, 'ne': 2, 'yes': 0, 'no': 2},
    'Czech': {'ano': 0, 'ne': 2, 'yes': 0, 'no': 2},
    'Danish': {'ja': 0, 'nej': 2, 'yes': 0, 'no': 2},
    'Dutch': {'ja': 0, 'nee': 2, 'yes': 0, 'no': 2},
    'English': {'yes': 0, 'no': 2},
    'Estonian': {'jah': 0, 'ei': 2, 'yes': 0, 'no': 2},
    'Finnish': {'kyllä': 0, 'ei': 2, 'yes': 0, 'no': 2},
    'French': {'oui': 0, 'non': 2, 'yes': 0, 'no': 2},
    'German': {'ja': 0, 'nein': 2, 'yes': 0, 'no': 2},
    'Greek': {'ναι': 0, 'όχι': 2, 'yes': 0, 'no': 2},
    'Haitian': {'wi': 0, 'non': 2, 'yes': 0, 'no': 2},
    'Hebrew': {'כן': 0, 'לא': 2, 'yes': 0, 'no': 2},
    'Hindi': {'हाँ': 0, 'नहीं': 2, 'yes': 0, 'no': 2},
    'Hungarian': {'igen': 0, 'nem': 2, 'yes': 0, 'no': 2},
    'Indonesian': {'ya': 0, 'tidak': 2, 'yes': 0, 'no': 2},
    'Italian': {'sì': 0, 'no': 2, 'yes': 0, 'no': 2},
    'Javanese': {'ya': 0, 'ora': 2, 'yes': 0, 'no': 2},
    'Japanese': {'はい': 0, 'いいえ': 2, 'yes': 0, 'no': 2},
    'Khmer': {'បាទ': 0, 'ទេ': 2, 'yes': 0, 'no': 2},
    'Korean': {'네': 0, '아니요': 2, 'yes': 0, 'no': 2},
    'Lao': {'ແມ່ນ': 0, 'ບໍ່': 2, 'yes': 0, 'no': 2},
    'Maithili': {'हाँ': 0, 'न': 2, 'yes': 0, 'no': 2},
    'Malay': {'ya': 0, 'tidak': 2, 'yes': 0, 'no': 2},
    'Malayalam': {'അതെ': 0, 'അല്ല': 2, 'yes': 0, 'no': 2},
    'Marathi': {'होय': 0, 'नाही': 2, 'yes': 0, 'no': 2},
    'Nepali': {'हो': 0, 'होइन': 2, 'yes': 0, 'no': 2},
    'Norwegian': {'ja': 0, 'nei': 2, 'yes': 0, 'no': 2},
    'Polish': {'tak': 0, 'nie': 2, 'yes': 0, 'no': 2},
    'Portuguese': {'sim': 0, 'não': 2, 'yes': 0, 'no': 2},
    'Quechua': {'arí': 0, 'mana': 2, 'yes': 0, 'no': 2},
    'Romanian': {'da': 0, 'nu': 2, 'yes': 0, 'no': 2},
    'Russian': {'да': 0, 'нет': 2, 'yes': 0, 'no': 2},
    'Serbian': {'да': 0, 'не': 2, 'yes': 0, 'no': 2},
    'Slovak': {'áno': 0, 'nie': 2, 'yes': 0, 'no': 2},
    'Slovenian': {'ja': 0, 'ne': 2, 'yes': 0, 'no': 2},
    'Spanish': {'sí': 0, 'no': 2, 'yes': 0, 'no': 2},
    'Swahili': {'ndiyo': 0, 'hapana': 2, 'yes': 0, 'no': 2},
    'Swedish': {'ja': 0, 'nej': 2, 'yes': 0, 'no': 2},
    'Tagalog': {'oo': 0, 'hindi': 2, 'yes': 0, 'no': 2},
    'Tamil': {'ஆம்': 0, 'இல்லை': 2, 'yes': 0, 'no': 2},
    'Telugu': {'అవును': 0, 'కాదు': 2, 'yes': 0, 'no': 2},
    'Thai': {'ใช่': 0, 'ไม่': 2, 'yes': 0, 'no': 2},
    'Tibetan': {'ཨེ': 0, 'མིན': 2, 'yes': 0, 'no': 2},
    'Turkish': {'evet': 0, 'hayır': 2, 'yes': 0, 'no': 2},
    'Ukrainian': {'так': 0, 'ні': 2, 'yes': 0, 'no': 2},
    'Urdu': {'ہاں': 0, 'نہیں': 2, 'yes': 0, 'no': 2},
    'Vietnamese': {'có': 0, 'không': 2, 'yes': 0, 'no': 2},
    'Zulu': {'yebo': 0, 'cha': 2, 'yes': 0, 'no': 2}
}

    nr_correct = sum(1 for x, y in zip(correct_answerlist, predicted_answerlist) if y and x == label_map[lang].get(y.lower(), 1))
    accuracy = round(100*(nr_correct / len(correct_answerlist)),1)

    return accuracy
    
def get_results_bnli(df,task,response_loc,lang):
    """
    Reads the response csv and calculates the accuracy for a model on a task.
    
    Parameters:
    df: orginial task English file with correct answer column.
    task: task name
    response_loc: string location of the response csv file.

    Returns:
    Accuracy score (%)
    """
    response = pd.read_csv(response_loc,sep=';',header=None)
    response.rename(columns={0:'response'},inplace=True)
    response = response.map(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)

    answer_list = []

    if task == 'bnli':

        for i in range(len(response)):
            # print(response.iloc[i,0])
            answer = extract_yesno_answer(response.iloc[i,0],lang)
            answer_list.append(answer)

        response['answer'] = answer_list
        # print(response)
        if task == 'bnli':

            df = df[df['label'] != 1]

        return calculate_accuracy_bnli(df,response,task,lang)