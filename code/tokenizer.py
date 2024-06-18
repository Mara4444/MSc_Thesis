from transformers import LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, BloomTokenizerFast, BloomForCausalLM
import transformers
import torch
import torch.nn as nn

model_name = "bigscience/bloomz-7b1-mt"
model = BloomForCausalLM.from_pretrained(model_name)
tokenizer = BloomTokenizerFast.from_pretrained(model_name)

# Armenian, Belarusian, Bulgarian, Burmese, Greek, 
# Hebrew, Japanese, Khmer, Korean, Lao, Serbian, Thai, Tibetan, Ukrainian, Russian 

print('BLOOMZ')

text = "ցի հիման վրա կազմեք թվային պատաս"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Armenian')
print(tokens) 
print(token_ids)  

text = "сфармулюйце лічбавы адказ"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Belarusian')
print(tokens) 
print(token_ids)  

text = "формулирайте числов отговор"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Bulgarian')
print(tokens) 
print(token_ids)  

text = "התבסס על השאלה, תבצע תשובה מספרית"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Hebrew')
print(tokens) 
print(token_ids)  

text = "အဖြေကို ဂဏန်းနဲ့ ရေးပါ။  အဖြေက"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Burmese')
print(tokens) 
print(token_ids)  

text = "διατυπώστε μια αριθμητική απάντηση"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Greek')
print(tokens) 
print(token_ids)  

text = "数値で答えを述べる"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Japanese')
print(tokens) 
print(token_ids)  

text = "សូមរៀបចំចម្លើយតាមចំនួន។"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Khmer')
print(tokens) 
print(token_ids)  

text = " 하여 숫자 로 답 을 정합 하십시오"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Korean')
print(tokens) 
print(token_ids)  

text = "ສ້າງຄໍາຕອບເປັນຕົວເລກ"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Lao')
print(tokens) 
print(token_ids)  

text = "формулишите нумерички одговор"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Serbian')
print(tokens) 
print(token_ids)  

text = "สร้างคําตอบด้วยตัวเลข"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Thai')
print(tokens) 
print(token_ids)  

text = "དྲི་བའི་གཞི་ནས་གྲངས་ཀྱི་ལན་འདེབས་ཤིག་བཀོད་རོགས།"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Tibetan')
print(tokens) 
print(token_ids)  

text = "можете дати за допомогою числа"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Ukrainian')
print(tokens) 
print(token_ids)  

text = "вопроса сформулируйте числовой ответ"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Russian')
print(tokens) 
print(token_ids)  

model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name,use_fast=True)
model = LlamaForCausalLM.from_pretrained(model_name)

# Arabic, Armenian, Bengali, Burmese, Greek, Hebrew, Hindi, 
# Khmer, Lao, Maithili, Malayalam, Marathi, Nepali, Tamil, Telugu, Thai, Tibetan, Urdu  

print('LLAMA-2')

text = " صياغة إجابة رقمية.  الجوا"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Arabic')
print(tokens) 
print(token_ids)  

text = "ցի հիման վրա կազմեք թվային պատաս"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Armenian')
print(tokens) 
print(token_ids)  

text = "একটি সংখ্যাসূচক উত্তর তৈরি করুন।  উত্তর"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Bengali')
print(tokens) 
print(token_ids)  

text = "အဖြေကို ဂဏန်းနဲ့ ရေးပါ။  အဖြေက"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Burmese')
print(tokens) 
print(token_ids) 

text = "διατυπώστε μια αριθμητική απάντηση"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Greek')
print(tokens) 
print(token_ids)  

text = "התבסס על השאלה, תבצע תשובה מספרית"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Hebrew')
print(tokens) 
print(token_ids)  

text = "សូមរៀបចំចម្លើយតាមចំនួន។"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Khmer')
print(tokens) 
print(token_ids)  

text = " संख्यात्मक उत्तर तैयार करें।"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Hindi')
print(tokens) 
print(token_ids)  

text = "ສ້າງຄໍາຕອບເປັນຕົວເລກ"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Lao')
print(tokens) 
print(token_ids)  

text = "สร้างคําตอบด้วยตัวเลข"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Thai')
print(tokens) 
print(token_ids)  

text = "དྲི་བའི་གཞི་ནས་གྲངས་ཀྱི་ལན་འདེབས་ཤིག་བཀོད་རོགས།"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Tibetan')
print(tokens) 
print(token_ids)  

text = "संख्यात्मक उत्तर तैयार करू।  उत्तर"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Maithili')
print(tokens) 
print(token_ids)  

text = "ഒരു സംഖ്യാ ഉത്തരം ഉണ്ടാക്കുക"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Malayalam')
print(tokens) 
print(token_ids)  

text = "आधारित, एक संख्यात्मक उत्तर तयार करा"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Marathi')
print(tokens) 
print(token_ids)  

text = "प्रश्नको आधारमा, संख्यात्मक उत्तर तयार पार्नुहोस्।"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Nepali')
print(tokens) 
print(token_ids)  

text = "கேள்விக்குரிய விடையை அடிப்படையாகக் கொண்"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Tamil')
print(tokens) 
print(token_ids)  

text = "ఆధారంగా, ఒక సంఖ్యా జవాబును రూపొందించండి"
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Telugu')
print(tokens) 
print(token_ids)  

text = "وال کی بنیاد پر، ایک عددی جواب تیار کریں. "
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('Urdu')
print(tokens) 
print(token_ids)  