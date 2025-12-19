import pandas as pd
import os
from pathlib import Path
import ollama
import tqdm
import json

PROJECT_ROOT = Path.cwd().parent.parent.parent
FIGURES_DIR = PROJECT_ROOT / 'figures'
DATA_DIR = PROJECT_ROOT / 'data' / 'newyorker_caption_contest'
JSON_DIR = DATA_DIR / 'contests.json'
CSV_DIR = DATA_DIR / 'data'
CSV_CLEAN_DIR = DATA_DIR / 'data_cleaned'
IMG_DIR = DATA_DIR / 'images'

CHANGE INPUT DIRECTORY

def store_joke_classification(classifications):
    for contest_id, captions in classifications.items():
        rows = []
        for cap_id, text in captions.items():
            
            # Extract numbers and labels
            parts = [p.strip() for p in text.split(',')]
            parsed = {}
            
            for part in parts:
                # Example: "0.6 Language"
                num, label = part.strip().split(' ', 1)
                parsed[label] = float(num)
            
            # build row
            row = {
                "caption_id": cap_id,
                **parsed
            }
            rows.append(row)

    
        pd.DataFrame(rows).to_csv('Data_augmented/'+contest_id+'.csv',index=False)
       
        
def _prompt_making_classifictation(cat,description=[]):
    '''
    This function creates the prompt for the catpion calssification.
    It explains 
    '''
    prompt=f"You are an expert in joke classification !\n"
    prompt+=f"Your job is to classify each joke to the following categories : {', '.join(cat)}.\n"
    prompt+=f"I want you to assert the confidence with which you're attributing the category. For example : 0.3 Langauage, 0.6 Logic, 0.0 Identity, 0.1 Action\n"
    prompt+=f"You answer really needs to be in the same structure as in the example !\n"
    prompt+=f"The definitions of the categories are :\n"
    prompt+=f"1) Language; Verbal humor, Allusion, Irony, Puns, Repartee, Wit, Ridicule\n"
    prompt+=f"2) Logic; Ideational humor, Absurdity, Coincidence, Conceptual surprise, Outwitting, Caught out, Misunderstanding\n"
    prompt+=f"3) Identity; Existential humor, Parody, Rigidity, Malicious pleasure, Condescension, Deceitful behavior, Self-deprecation\n"
    prompt+=f"4) Action; Physical/ non-verbal humour, Peculiar face, Peculiar music, Clumsiness, Repulsive behavior\n"
    if description:
        prompt+=f"Use the image description as well as the uncanny image discription to assert the caption category :\n "
        prompt+=f"Description; {description[0]}\n"
        prompt+=f"Uncanny description; {description[1]}\n"
    return prompt

def check_done(path,compare=[]):
    done=[]
    for classified in os.listdir(path):
        done.append(classified.split('/')[-1].removesuffix('.csv'))
    if not compare:
        for contest in os.listdir(str(CSV_CLEAN_DIR)):
            compare.append(contest.split('/')[-1].removesuffix('.csv'))
    missing = [c for c in compare if c not in done]
    return missing
    
def classify_captions(API_key,image_description=False,save=False,to_be_done_id=check_done('Data_augmented')):
    '''
    Iterates trhough all the contests, and attribute the captions into one from the Berger humor categories.
    It does so using ollama with a specific model. In order to use this function, the user needs to make an Ollama account,
    and created an API key.
    API_key: User ollam API key in str format
    image_descriptiom: bool, if true it adds the image description in the prompt
    save: bool, if true saves the results at each contest iteration
    to_be_done_id: list of contest ids to classify, the base values are computed by iterating the on the already done classified 
    contests, and all the contest. If a contest is given in the list but has already been classified, the function will pass the 
    specific contest.
    
    returns a dict with the contest id as key and a data frame with the results and the caption id for later manipulation.
    '''
    
    cat=['Language','Logic','Identity','Action'] # Berger Humor categories
    # in case the image description is needed
    file_path = str(JSON_DIR)
    with open(file_path, "r") as file:
        contests = json.load(file)
    contests_df = pd.DataFrame(contests)
    
    # Dict to store the data
    contest_class={}
    
    # Initialise the Client
    os.environ['OLLAMA_API_KEY'] = API_key
    client = ollama.Client(host='https://ollama.com',headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')})
    
    #Check the contest to classify in oreder to avoid computing already classified contest
    to_be_done_id=check_done('Data_augmented',to_be_done_id)
    print(len(to_be_done_id),' contest left')
    
    
    for contest in to_be_done_id: # Iterates through contests
        df=pd.read_csv(str(CSV_CLEAN_DIR)+'/'+contest+'.csv',engine="python", on_bad_lines="warn")
        
        # Sub sammple
        length=df.shape[0]
        df1_idx=list(df.iloc[:25,:].index)
        df2_idx=list(df.iloc[(length-25):,:].index)
        index=df1_idx+df2_idx
        
        # In case if image description is True, gets the image description of the contest, if the argument is false, gives an empty
        # list to the prompt making function
        descrip=[]
        if image_description:
            contest_json=contests_df[contests_df['contest_id']==contest]
            descrip=[contest_json['metadata'].iloc[0].get('image_descriptions'),contest_json['metadata'].iloc[0].get('image_uncanny_descriptions')]
            
        # Make the prompt
        prompt=_prompt_making_classifictation(cat,descrip)
        
        # prepare dict to store the contest results
        results = {}
        print('======================'+contest+'======================')
        for idx in tqdm(index): # iterates trhough the subsampled list of caption ids
            row=df.iloc[idx,:]
            prompt_cap=prompt + f'Classify this caption : ' + row['caption']
            
            messages = [
                {
                    'role': 'user',
                    'content': prompt_cap,
                },
                ]
            full_response = ''
            # Stream the response and accumulate it
            for part in client.chat('gpt-oss:120b', messages=messages, stream=True): #
                if part.message.content:
                    content = part.message.content
                    full_response += content
                    #print(content, end='', flush=True)
            # Store the complete response
            results[str(idx)] = full_response
        contest_class[str(contest)]=results
        if save:
            store_joke_classification(contest_class)
    return contest_class

def correct_format_LLMS(input_dir):
    for csv_file in os.listdir(input_dir):
        if not csv_file.endswith("cleaned.csv"):
            continue

        path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(path)


        # ---- 1. Keep only the last 50 rows ----
        df = df.iloc[-50:].copy()
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        # ---- 2. Rename columns properly ----
        if not df.shape[1]==5: # Correct number of columns
            df.columns = [
                "caption_id",
                "Language_1",
                "Logic",
                "Identity",
                "Action",
                "Language_2"]

            # ---- 3. Merge duplicated Language columns ----
            df["Language"] = df["Language_1"].combine_first(df["Language_2"])

            # ---- 4. Drop old duplicated columns ----
            df = df.drop(columns=["Language_1", "Language_2"])

        # ---- 5. Ensure numeric columns are numeric ----
        for col in ["Language", "Logic", "Identity", "Action"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
       
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df.to_csv(input_dir+'/'+csv_file,index=False)


def add_observables_LLMS(input_dir):
    all_dfs = []
    for csv_file in os.listdir(input_dir):
        if not csv_file.endswith('.csv'):
            continue
        
        path = os.path.join(input_dir, csv_file)
        df = pd.read_csv(path)
        
        clean_path = os.path.join(str(CSV_CLEAN_DIR), csv_file)
        real_df = pd.read_csv(clean_path)
        
        df['mean'] = real_df.iloc[df['caption_id'].values]['mean'].values
        
        contest_id = csv_file.replace('_cleaned.csv', '').replace('.csv', '')
        df['contest_id'] = contest_id
        
        df['dominant_category'] = df[['Logic', 'Identity', 'Action', 'Language']].idxmax(axis=1)

        # Identify the value of the dominant category
        df['dominant_value'] = df[['Logic', 'Identity', 'Action', 'Language']].max(axis=1)
        df_filtered = df[df['dominant_value'] > 0.25].dropna()  # errors in chatgpt

        all_dfs.append(df_filtered)
        
    super_df = pd.concat(all_dfs, ignore_index=True)
    super_df.to_csv('Data_augmented/super.csv', index=False)
    return super_df

