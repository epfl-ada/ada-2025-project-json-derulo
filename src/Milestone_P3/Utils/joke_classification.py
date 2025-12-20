import pandas as pd
import os
from pathlib import Path
import ollama
import tqdm
import json

def store_joke_classification(classifications,input_data_dir):
    '''
    Store in scv file the answer from the LLM. 
    classificationss : dict that has the contest ids as key, then another dict with the captions ids as keys and the LLM answer 
    input_data_dir : where to store the csvs
    
    returns nothing
    '''
    for contest_id, captions in classifications.items():
        #Perpare to store the data
        rows = []
        for cap_id, text in captions.items():
            # Deconstruct the LLM answer 
            parts = [p.strip() for p in text.split(',')]
            parsed = {}
            
            # Deconstruct the category and it proportion
            for part in parts:
                num, label = part.strip().split(' ', 1)
                parsed[label] = float(num)
            
            # build row of future df
            row = {'caption_id': cap_id,**parsed}
            rows.append(row)

        #saves the results in TODO
        pd.DataFrame(rows).to_csv(input_data_dir+'/'+contest_id+'.csv',index=False)
       
        
def _prompt_making_classifictation(cat,description=[]):
    '''
    This function creates the prompt for the catpion calssification.
    It explains what are the Berger Humor categories and specify how the answer needs to be formated
    cat : list of the humor categories,
    descritpion: if not empty list, add to the caption the image descritption and the uncanny description.
    
    returns the prompt
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
    '''
    Compares the files csv files in path and compare, and removes the one in comonn
    path : str path were the already done csv are stored
    compare: if not given, compares with all the data in CSV_CLEAN_DIR
    
    returns a filter list of contest to classify
    '''
    done=[]
    for classified in os.listdir(path):
        done.append(classified.split('/')[-1].removesuffix('.csv'))
    if not compare:
        for contest in os.listdir('../../../data/newyorker_caption_contest/cleaned_data/'):
            compare.append(contest.split('/')[-1].removesuffix('.csv'))
    missing = [c for c in compare if c not in done]
    return missing
    
def classify_captions(API_key,input_data_dir,image_description=False,save=False,to_be_done_id=[]):
    '''
    Iterates trhough all the contests, and attribute the captions into one from the Berger humor categories.
    It does so using ollama with a specific model. In order to use this function, the user needs to make an Ollama account,
    and created an API key.
    API_key: User ollam API key in str format
    input
    image_descriptiom: bool, if true it adds the image description in the prompt
    save: bool, if true saves the results in a csv at each contest iteration
    to_be_done_id: list of contest ids to classify. If a contest is given in the list but has already been classified, 
    the function will pass the specific contest.
    
    returns a dict with the contest id as key and a data frame with the results and the caption id for later manipulation.
    '''
    input_dir_class=str(input_dir_class)
    cat=['Language','Logic','Identity','Action'] # Berger Humor categories
    # in case the image description is needed
    file_path = str(input_data_dir+'contests.json')
    with open(file_path, "r") as file:
        contests = json.load(file)
    contests_df = pd.DataFrame(contests)
    
    # Dict to store the data
    contest_class={}
    
    # Initialise the Client
    os.environ['OLLAMA_API_KEY'] = API_key
    client = ollama.Client(host='https://ollama.com',headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')})
    
    #Check the contest to classify in oreder to avoid computing already classified contest
    to_be_done_id=check_done(input_data_dir+'/LLM_classifications/',to_be_done_id)
    print(len(to_be_done_id),' contest left')
    
    
    for contest in to_be_done_id: # Iterates through contests
        df=pd.read_csv(input_data_dir+'/LLM_classifications/'+contest+'.csv',engine='python', on_bad_lines='warn')
        
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
            
            # Create the message in the correct format for ollams
            messages = [{'role': 'user','content': prompt_cap,},]
            
            # prepare to store the LLM answer
            full_response = ''
            
            # Submit the prompt to the LLM
            for part in client.chat('gpt-oss:120b', messages=messages, stream=True): #
                if part.message.content:
                    content = part.message.content
                    full_response += content
            #store the anwer with the caption id
            results[str(idx)] = full_response
            
        # store all the classified caption with the contest id
        contest_class[str(contest)]=results
        
        #saves in csv if specified
        if save:
            store_joke_classification(contest_class,input_data_dir+'/LLM_classifications')
    return contest_class

def correct_format_LLMS(input_dir):
    '''
    Because of some errors in the coding (that were corrected), and the errors of the wrogly formated ansers of the LLM, This functions
    corrects the mistakes e.g new columns empty columns, two identical columns, etc And overwrite the csvs
    input_dir: directory where the csv are stored
    
    returns nothing
    '''
    input_dir=str(input_dir)
    for csv_file in os.listdir(input_dir): # iterates
        if not csv_file.endswith('cleaned.csv'): # presence of super csv
            continue

        df = pd.read_csv(input_dir+'/'+csv_file)

        # keep only the last 50 rows
        df = df.iloc[-50:].copy()
        
        # removes empty columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # rename columns properly 
        if not df.shape[1]==5: # Correct number of columns
            df.columns = ['caption_id','Language_1','Logic','Identity','Action','Language_2']

            # merge duplicated language columns
            df['Language'] = df['Language_1'].combine_first(df['Language_2'])

            # drop duplicated columns
            df = df.drop(columns=['Language_1', 'Language_2'])

        # ensure the values are numeric  
        for col in ['Language', 'Logic', 'Identity', 'Action']:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        # removes empty columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # overwrite the csvs
        df.to_csv(input_dir+'/'+csv_file,index=False)


def add_observables_LLMS(input_data_dir):
    '''   
    Add obersevables to the csv such as the mean of the caption for later analisys, the contest id, the dominant category as well as its
    dominant value. Drops the rows with dominant values < 0.25 because all the the categories would be then 'dominant'. Then regroup all
    the csv and creates one with all the informations of all contests 
    input_dir: directory where the csv are stored
    
    returns df from super.csv with all the data
    '''
    input_data_dir=str(input_data_dir)
    class_dir=input_data_dir+'/LLM_classifications/'
    #store results
    all_dfs = []
    
    for csv_file in os.listdir(class_dir):
        if not csv_file.endswith('cleaned.csv'): # check of the files
            continue
        
        df = pd.read_csv(class_dir + csv_file)
        
        real_df = pd.read_csv(input_data_dir+'/cleaned_data/'+csv_file)
        
        # add the means
        df['mean'] = real_df.iloc[df['caption_id'].values]['mean'].values
        
        # add the contest id
        contest_id = csv_file.replace('_cleaned.csv', '').replace('.csv', '')
        df['contest_id'] = contest_id
        
        # add the dominant category
        df['dominant_category'] = df[['Logic', 'Identity', 'Action', 'Language']].idxmax(axis=1)

        # add the value of the dominant value
        df['dominant_value'] = df[['Logic', 'Identity', 'Action', 'Language']].max(axis=1)
        df_filtered = df[df['dominant_value'] > 0.25].dropna()  # errors in LLM

        all_dfs.append(df_filtered)
    
    # creates the super df with all the data and store it    
    super_df = pd.concat(all_dfs, ignore_index=True)
    super_df.to_csv(class_dir+'super.csv', index=False)
    return super_df

