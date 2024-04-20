import yaml

def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as e:
            print("Error loading YAML:", e)
            return None

# Example usage
if __name__ == "__main__":
    config = load_config("config.yaml")
    tokenizer_name = config['tokenizer_name']
    