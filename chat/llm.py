import os
import yaml


def read_yaml_to_dict(path: str) -> dict:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    prompts_file = os.path.join(current_directory, path)
    with open(prompts_file, 'r') as file:
        yaml_dict = yaml.safe_load(file)
    return yaml_dict


def generate_prompt_template(
        question: str,
        context: str,
        chat_category_name: str, 
        yaml_dict: dict
        ) -> dict:
    prompts = [
        prompt for prompt in yaml_dict['prompts'] \
        if chat_category_name in prompt['name']
    ]
    system_message = prompts[0]['system']
    message_template = \
        prompts[0]['task'] + "\nQuestion: {question}\n\nContext:\n{context}\n\nHelpful Answer:"    
    user_message = message_template.format(question=question, context=context)

    return user_message, system_message


def openai_answer(client, user_message, system_message):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            ],
            temperature=0,
            # max_tokens=64,
            # top_p=1
        )
    return response.choices[0].message.content