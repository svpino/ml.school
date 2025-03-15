from llama_cpp import Llama
import json

def get_backend_class():
    # Load local config
    with open('local.json', 'r') as f:
        config = json.load(f)
    
    # Initialize local backend
    model = Llama(
        model_path=config['model_path'],
        n_ctx=config.get('n_ctx', 2048),
        n_threads=config.get('n_threads', 4)
    )
    
    # Get backend class name
    backend_name = model.__class__.__name__
    return backend_name

if __name__ == '__main__':
    backend_class = get_backend_class()
    print(f"Backend class name: {backend_class}") 