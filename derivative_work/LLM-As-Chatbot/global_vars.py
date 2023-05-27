import yaml
from transformers import GenerationConfig
from models import vicuna

def initialize_globals(args):
    global model, model_type, stream_model, tokenizer
    global gen_config, gen_config_raw    
    global gen_config_summarization
    
    model_type_tmp = "zenai"

    print(f"determined model type: {model_type_tmp}")        

    try:
        if model is not None:
            del model

        if tokenizer is not None:
            del tokenizer
    except NameError:
        pass

    load_model = get_load_model(model_type_tmp)
    model, tokenizer = load_model()        
        
    gen_config, gen_config_raw = get_generation_config(args.gen_config_path)
    gen_config_summarization, _ = get_generation_config(args.gen_config_summarization_path)
    model_type = model_type_tmp
    stream_model = model
        
def get_load_model(model_type):
    return vicuna.load_model
    
def get_generation_config(path):
    with open(path, 'rb') as f:
        generation_config = yaml.safe_load(f.read())
        
    generation_config = generation_config["generation_config"]

    return GenerationConfig(**generation_config), generation_config

def get_constraints_config(path):
    with open(path, 'rb') as f:
        constraints_config = yaml.safe_load(f.read())
        
    return ConstraintsConfig(**constraints_config), constraints_config["constraints"]
