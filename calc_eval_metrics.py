
from src.eval_metrics import * 
import pandas as pd 

# Example sentences 
TEST_ORIG_LIST = [
    "The weather is nice today.",  # English
    "The weather is nice today.",  # English
    "The weather is nice today.",  # English
    "The weather is nice today.",  # English
    "هذا الطقس جيد اليوم",  # Arabic
    "هذا الطقس جيد اليوم",  # Arabic
    "هذا الطقس جيد اليوم",  # Arabic
    "هذا الطقس جيد اليوم",  # Arabic
    "El clima está agradable hoy.",  # Spanish
    "El clima está agradable hoy.",  # Spanish
    "El clima está agradable hoy.",  # Spanish
    "El clima está agradable hoy.",  # Spanish
    "Le temps est agréable aujourd'hui.",  # French
    "Le temps est agréable aujourd'hui.",  # French
    "Le temps est agréable aujourd'hui.",  # French
    "Le temps est agréable aujourd'hui.",  # French
    "Das Wetter ist heute schön.",  # German
    "Das Wetter ist heute schön.",  # German
    "Das Wetter ist heute schön.",  # German
    "Das Wetter ist heute schön."  # German
]
TEST_PP_LIST = [
    "Oh boy, what a fine movie.",  # Very close to original, same language
    "The weather is terrible today.",  # Contradiction, same language
    "Weather nice is today the.",  # Grammatically not fluent, same language
    "The climate is gut aujourd'hui.",  # Neutral, same language
    "هذا الطقس جيد اليوم، صحيح؟",  # Very close to original, same language
    "هذا الطقس سيء اليوم",  # Contradiction, same language
    "جيد الطقس هذا اليوم",  # Grammatically not fluent, same language
    "The weather is good today.",  # Neutral, different language
    "El clima está agradable hoy, ¿verdad?",  # Very close to original, same language
    "El clima está terrible hoy.",  # Contradiction, same language
    "Está el clima agradable hoy.",  # Grammatically not fluent, same language
    "The weather is good today.",  # Neutral, different language
    "Le temps est agréable aujourd'hui, n'est-ce pas?",  # Very close to original, same language
    "Le weather est terrible aujourd'hui.",  # Contradiction, same language
    "Aujourd'hui est le temps agréable.",  # Grammatically not fluent, same language
    "Das Wetter ist heute schön.",  # Neutral, different language
    "Das Wetter ist heute schön, oder?",  # Very close to original, same language
    "Das Wetter ist heute schrecklich.",  # Contradiction, same language
    "Heute ist das Wetter schön.",  # Grammatically not fluent, same language
    "The weather is nice today."  # Neutral, different language
]
TEST_LANGS = [
    'en', 'en', 'en', 'en', 'ar', 'ar', 'ar', 'ar', 'es', 'es', 'es', 'es', 'fr', 'fr', 'fr', 'fr', 'de', 'de', 'de', 'de'
]


### TEST CODE 
# SETUP 
device = 'cuda:0'
scorer_d = set_up_all_scorers(device)

# filtering step here to only get successful examples in df 
df = pd.DataFrame({'orig': TEST_ORIG_LIST, 'pp': TEST_PP_LIST, 'lang': TEST_LANGS})
df['flip'] =  1  # dummy value for now 

res_df = score_df(df, scorer_d)