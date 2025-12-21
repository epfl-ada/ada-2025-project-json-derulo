from tqdm import tqdm
import ollama
import random
import base64

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
        image_path=root+'/'+str(complete['contest_id'])+'.jpg'
        with open(image_path, "rb") as f:
                #image = f.read() #test 1
                #image =base64.b64encode(f.read()).decode("utf-8") # test2 
                image=image_path #test3
        meta = complete['metadata']
        
        examples.append({ 
            'image_path': image, 'image_loc': meta.get('image_locations'),
            'image_descri': meta.get('image_descriptions'), 'image_uncan_descri':meta.get('image_uncanny_descriptions'),
            'images_entities': meta.get('entities'),'images_questions': meta.get('questions')
        })
    return examples

def _prompt_making_description(examples):
    prompt = "You are an expert that generates detailed and structured metadata for images.\n\n"
    prompt+=f'Next are some examples for different images, I want your answer to be in the same structure as in the examples\n'    
    for example in examples:  # Fewer examples for smaller models
        prompt += f"Example:\n"
        #prompt += f"image_path: {example['image_path']}\n" # or image_path
        prompt += f"Location: {example['image_loc']}\n"
        prompt += f"Description: {example['image_descri']}\n"
        prompt += f"Uncanny description : {example['image_uncan_descri']}\n"
        prompt += f"Entities: {example['images_entities']}\n"
        prompt += f"Questions: {example['images_questions']}\n\n"
    prompt+='Now to our image !'
    return prompt

def add_image_descriptions(contests_df, root): 
    '''
    Adds the missing metada related to the image
    
    contests_df: contests dataframe from the Json 
    
    root: Path or str that points the folder where the images are
    
    Returns the metadata, with the id contest, of those who lacked them. 
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
        "image_locations : what location is the image decripting", "image_descriptions : What is happening in the image", "image_uncanny_descriptions : what is special about the image situation",
        "entities: wikipedias links to the entities from the image", "questions: what questions can be asked from the image"," your answer confidence: (between 0-1)"
    ]
    results = []
    for ids in tqdm(missing_df_ids):
        
        meta = contests_df.loc[contests_df['contest_id'] == ids, 'metadata'].iloc[0]
        if not isinstance(meta, dict):
            meta = {}
        image_path=root+'/'+str(ids)+'.jpg'
        examples = _examples_for_ai(complete_df,root,number_of_ex=3)    
        prompt=_prompt_making_description(examples)
        with open(image_path, "rb") as f:
                #image = f.read() test 1
                image =base64.b64encode(f.read()).decode("utf-8") #test 2
                #image=image_path #test3
        #prompt = "You are an expert that generates detailed and structured metadata for images.\n\n"
        prompt += f"""
        Fill in the missing keys: {', '.join(required_keys)}, for the image.
        """
        print(f'id : {ids}, prompt {prompt}')
        try:
            response = ollama.generate(
                model='llava:7b',  # 'llava:7b'
                prompt=prompt,# + image,
                images=[image],
                options={'temperature': 0.1}  # control haluciantion low temp mostly consistent, slight creativity
            )
            description = response['response']
        
            results.append({
            'contest_id': ids,
            'description': description
            })
        except Exception as e:
            print(f"Error processing contest {ids}: {e}")
        break
    return results


