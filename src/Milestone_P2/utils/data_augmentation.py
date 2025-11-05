from tqdm import tqdm
import ollama
import random


def _missing_meta(meta, what):
    """
    Check if a single metadata dictionary is missing a given key.
    meta: dictionary that is supposed to contain all the metadata
    what: key that we are looking for
    """
    return (
        not isinstance(meta, dict)
        or what not in meta
        )
    

def ids_missing_key(df, what):
    """
    Returns the contest IDs (not indices) for which the metadata dictionary is missing the key `what`.

    Args:
        metadata_list (list): List of metadata dicts (contests_df['metadata'])
        what (str): Key to check for (e.g. 'num_captions', 'image_descriptions', etc.)
        contest_ids (list or Series): List of contest IDs aligned with metadata_list
    """
    return [
        int(df['contest_id'][i])
        for i, meta in enumerate(df['metadata'])
        if _missing_meta(meta, what)
    ]

def _examples_for_ai(df,root,number_of_ex=5):
    examples = []
    for _, complete in random.sample(list(df.iterrows()),number_of_ex):
        image_path=root+'images/'+str(complete['contest_id'])+'.jpg'
        #with open(image_path, "rb") as f:
                #image_b64 = base64.b64encode(f.read()).decode("utf-8")
        meta = complete['metadata']
        
        examples.append({
            'image': image_path,'image_loc': meta.get('image_locations'),
            'image_descri': meta.get('image_descriptions'), 'image_uncan_descri':meta.get('image_uncanny_descriptions'),
            'images_entities': meta.get('entities'),'images_questions': meta.get('questions')
        })
    return examples

def _prompt_making(examples):
    prompt = "You are an assistant that generates detailed and structured metadata for images.\n\n"
        
    for example in examples:  # Fewer examples for smaller models
        prompt += f"Example:\n"
        prompt += f"image_path: {example['image']}\n"
        prompt += f"Location: {example['image_loc']}\n"
        prompt += f"Description: {example['image_descri']}\n"
        prompt += f"Uncanny description : {example['image_uncan_descri']}\n"
        prompt += f"Entities: {example['images_entities']}\n"
        prompt += f"Questions: {example['images_questions']}\n\n"
    
    return prompt

def add_image_descriptions(contests_df, root): 
    '''
    contests_df: contests dataframe from the Json 
    
    root: Path or str that points the folder where the images are
    ''' 
    root=str(root)
    # Test if Ollama is running and pull a model
    try:
        ollama.list()
    except:
        print("Please install and start Ollama first: https://ollama.ai/")
        return contests_df
    missing_df_ids = ids_missing_key(contests_df,'image_descriptions')
    complete_df = contests_df[~contests_df['contest_id'].isin(missing_df_ids)] ## removes the missings dfs

    required_keys = [
        "image_locations", "image_descriptions", "image_uncanny_descriptions",
        "entities", "questions",
    ]
    results = []
    for ids in tqdm(missing_df_ids):
        
        meta = contests_df.loc[contests_df['contest_id'] == ids, 'metadata'].iloc[0]
        if not isinstance(meta, dict):
            meta = {}
        image_path=root+'images/'+str(ids)+'.jpg'
        examples = _examples_for_ai(complete_df,root)    
        prompt=_prompt_making(examples)
        prompt += f"""
        Incomplete metadata: {meta}
        Fill in the missing keys: {', '.join(required_keys)} for the following image {image_path}
        Provide a JSON object with all fields completed.
        Return ONLY valid JSON, no other text.
        """
        try:
            response = ollama.generate(
                model='llava:7b',  # or 'mistral', 'codellama', etc.
                prompt=prompt,
                options={'temperature': 0.1}  # control randomness/creativity, low temp mostly consistent, slight creativity
            )
            description = response['response']
        
            results.append({
            'contest_id': ids,
            'description': description
            })
        except Exception as e:
            print(f"Error processing contest {ids}: {e}")
            continue
    return contests_df


