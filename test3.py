import requests
import json

# REQUIRES 3 FILES IN THE WORKING DIRECTORY
# prompt.txt - contains baseline prompt for messages sent
# input.txt - contains line-separated inputs to be sent after prompt.txt
# out.txt - (will be created) contains the output from this program
# example: python test3.py

# For local streaming, the websockets are hosted without ssl - http://
# change HOST to YOUR oobabooga server's api
HOST = 'localhost:5000'
URI = f'http://{HOST}/api/v1/generate'

# For reverse-proxied streaming, the remote will likely host with ssl - https://
# URI = 'https://your-uri-here.trycloudflare.com/api/v1/generate'


PROGRESS_FILE = 'progress.txt'


def extract_last_output(prompt_text):
    reversed_lines = prompt_text.split('\n')[::-1]
    for line in reversed_lines:
        if line.strip().startswith('OUTPUT:'):
            return line.strip()[len('OUTPUT:'):]
    return None


def save_progress(line_number):
    with open(PROGRESS_FILE, 'w') as progress_file:
        progress_file.write(str(line_number))


def get_progress():
    try:
        with open(PROGRESS_FILE, 'r') as progress_file:
            return int(progress_file.read())
    except FileNotFoundError:
        return 0


def is_valid_json(json_string):
    try:
        json_object = json.loads(json_string)
        return isinstance(json_object, dict)
    except ValueError:
        return False


def run(prompt, input_lines):
    last_processed_line = get_progress()

    with open('out.txt', 'a') as output_file:
        with open('prompt.txt', 'r') as prompt_file:
            prompt_text = prompt_file.read().strip()

        for i, line in enumerate(input_lines, start=1):
            if i <= last_processed_line:
                continue


            request = {
                'prompt': prompt_text + line,
                'max_new_tokens': 1000,
                'preset': 'None',
                'do_sample': True,
                'temperature': 0.4,
                'top_p': 0.1,
                'typical_p': 0.5,
                'epsilon_cutoff': 0,  # In units of 1e-4
                'eta_cutoff': 0,  # In units of 1e-4
                'tfs': 1,
                'top_a': 0,
                'repetition_penalty': 1.18,
                'repetition_penalty_range': 0,
                'top_k': 40,
                'min_length': 0,
                'no_repeat_ngram_size': 0,
                'num_beams': 1,
                'penalty_alpha': 0,
                'length_penalty': 1,
                'early_stopping': False,
                'mirostat_mode': 0,
                'mirostat_tau': 5,
                'mirostat_eta': 0.1,

                'seed': -1,
                'add_bos_token': True,
                'truncation_length': 4096,
                'ban_eos_token': False,
                'skip_special_tokens': True,
                'stopping_strings': []
            }


            response = requests.post(URI, json=request)

            if response.status_code == 200:
                result = response.json()['results'][0]['text']
                full_output = prompt_text + line + result
                trimmed_output = extract_last_output(full_output)
                if trimmed_output:
                    print(full_output)
                    print(trimmed_output)  # Print only last occurrence of 'OUTPUT:'
                    if is_valid_json(trimmed_output):
                        output_file.write(trimmed_output + '\n')
                        output_file.flush()  # Flush buffer to write as we go
                    else:
                        print("Invalid JSON response. Retrying...")
                        continue  # Retry request for valid response

            save_progress(i)


if __name__ == '__main__':
    prompt = "In order to make homemade bread, follow these steps:\n1)"
    with open('input.txt', 'r') as input_file:
        input_lines = input_file.readlines()

    run(prompt, input_lines)