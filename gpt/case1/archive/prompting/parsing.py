import pandas as pd 

list_uk_us_df = pd.read_csv('list_uk_us.csv')
list_uk_us = list_uk_us_df.to_dict('split', index=False)
    #[("colour", 'color'),]
 
             
id2contexts_df = pd.read_csv('id2contexts.csv')
id2contexts = id2contexts_df.to_dict('split', index=False)
   #Current: [[0, 'raw', 'My favorite color is red.'],[1, 'raw', 'I will have some cookies as a snack.']]
   #Adjust to: 1: {'raw': 'My favorite color is red.',}



for i, text_obj in id2contexts.items():
   uk_text = text_obj['raw']
   us_text = text_obj['raw']
   for uk, us in list_uk_us:
       uk_text = uk_text.replace(us, uk)
       us_text = us_text.replace(uk, us)
   text_obj.update({
       'uk': uk_text,
       'us': us_text,
   })
