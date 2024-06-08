import os
import sys

sys.path.insert(0, os.getcwd())
import argparse
import re
import time

import pandas
import numpy as np
from tqdm import tqdm
import random
import gradio as gr
import json
from utils import combine_audio, save_audio, batch_split, normalize_zh
from tts_model import load_chat_tts_model, clear_cuda_cache, deterministic, generate_audio_for_seed

parser = argparse.ArgumentParser(description="Gradio ChatTTS MIX")
parser.add_argument("--source", type=str, default="huggingface", help="Model source: 'huggingface' or 'local'.")
parser.add_argument("--local_path", type=str, help="Path to local model if source is 'local'.")
parser.add_argument("--share", default=False, action="store_true", help="Share the server publicly.")

args = parser.parse_args()

# Directory to store audio seed files
SAVED_DIR = "saved_seeds"

# Create directory if it doesn't exist
if not os.path.exists(SAVED_DIR):
    os.makedirs(SAVED_DIR)

# File path
SAVED_SEEDS_FILE = os.path.join(SAVED_DIR, "saved_seeds.json")

# Selected seed index
SELECTED_SEED_INDEX = -1

# Initialize JSON file
if not os.path.exists(SAVED_SEEDS_FILE):
    with open(SAVED_SEEDS_FILE, "w") as f:
        f.write("[]")

chat = load_chat_tts_model(source=args.source, local_path=args.local_path)
# chat = None
# chat = load_chat_tts_model(source="local", local_path=r"models")

# Maximum number of audio components
max_audio_components = 10

def load_seeds():
    with open(SAVED_SEEDS_FILE, "r") as f:
        global saved_seeds
        saved_seeds = json.load(f)
    return saved_seeds

def display_seeds():
    seeds = load_seeds()
    # Convert to List[List] format
    return [[i, s['seed'], s['name']] for i, s in enumerate(seeds)]

saved_seeds = load_seeds()
num_seeds_default = 2

def save_seeds():
    global saved_seeds
    with open(SAVED_SEEDS_FILE, "w") as f:
        json.dump(saved_seeds, f)
    saved_seeds = load_seeds()

# Add seed
def add_seed(seed, name, save=True):
    for s in saved_seeds:
        if s['seed'] == seed:
            return False
    saved_seeds.append({
        'seed': seed,
        'name': name
    })
    if save:
        save_seeds()

# Modify seed
def modify_seed(seed, name, save=True):
    for s in saved_seeds:
        if s['seed'] == seed:
            s['name'] = name
            if save:
                save_seeds()
            return True
    return False

def delete_seed(seed, save=True):
    for s in saved_seeds:
        if s['seed'] == seed:
            saved_seeds.remove(s)
            if save:
                save_seeds()
            return True
    return False

def generate_seeds(num_seeds, texts, tq):
    """
    Generate random audio seeds and save them
    :param num_seeds:
    :param texts:
    :param tq:
    :return:
    """
    seeds = []
    sample_rate = 24000
    # Split text by line and normalize numbers and punctuation characters
    texts = [normalize_zh(_) for _ in texts.split('\n') if _.strip()]
    print(texts)
    if not tq:
        tq = tqdm
    for _ in tq(range(num_seeds), desc=f"Random voice color generation in progress..."):
        seed = np.random.randint(0, 9999)

        filename = generate_audio_for_seed(chat, seed, texts, 1, 5, "[oral_2][laugh_0][break_4]", 0.3, 0.7, 20)
        seeds.append((filename, seed))
        clear_cuda_cache()

    return seeds

# Save selected audio seed
def do_save_seed(seed):
    seed = seed.replace('Save seed ', '').strip()
    if not seed:
        return
    add_seed(int(seed), seed)
    gr.Info(f"Seed {seed} has been saved.")

def do_save_seeds(seeds):
    assert isinstance(seeds, pandas.DataFrame)

    seeds = seeds.drop(columns=['Index'])

    # Convert DataFrame to dictionary list format and convert keys to lowercase
    result = [{k.lower(): v for k, v in row.items()} for row in seeds.to_dict(orient='records')]
    print(result)
    if result:
        global saved_seeds
        saved_seeds = result
        save_seeds()
        gr.Info(f"Seeds have been saved.")
    return result

def do_delete_seed(val):
    # Extract index from val using regex
    index = re.search(r'(\d+)(\d+)', val)
    global saved_seeds
    if index:
        index = int(index.group(1))
        seed = saved_seeds[index]['seed']
        delete_seed(seed)
        gr.Info(f"Seed {seed} has been deleted.")
    return display_seeds()

def seed_change_btn():
    global SELECTED_SEED_INDEX
    if SELECTED_SEED_INDEX == -1:
        return 'Delete'
    return f'Delete idx=[{SELECTED_SEED_INDEX[0]}]'

def audio_interface(num_seeds, texts, progress=gr.Progress()):
    """
    Generate audio
    :param num_seeds:
    :param texts:
    :param progress:
    :return:
    """
    seeds = generate_seeds(num_seeds, texts, progress.tqdm)
    wavs = [_[0] for _ in seeds]
    seeds = [f"Save seed {_[1]}" for _ in seeds]
    # Empty parts
    all_wavs = wavs + [None] * (max_audio_components - len(wavs))
    all_seeds = seeds + [''] * (max_audio_components - len(seeds))
    return [item for pair in zip(all_wavs, all_seeds) for item in pair]

def audio_interface_empty(num_seeds, texts, progress=gr.Progress(track_tqdm=True)):
    return [None, ""] * max_audio_components

def update_audio_components(slider_value):
    # Update Audio component visibility based on slider value
    k = int(slider_value)
    audios = [gr.Audio(visible=True)] * k + [gr.Audio(visible=False)] * (max_audio_components - k)
    tbs = [gr.Textbox(visible=True)] * k + [gr.Textbox(visible=False)] * (max_audio_components - k)
    print(f'k={k}, audios={len(audios)}')
    return [item for pair in zip(audios, tbs) for item in pair]

def seed_change(evt: gr.SelectData):
    # print(f"You selected {evt.value} at {evt.index} from {evt.target}")
    global SELECTED_SEED_INDEX
    SELECTED_SEED_INDEX = evt.index
    return evt.index

def generate_tts_audio(text_file, num_seeds, seed, speed, oral, laugh, bk, min_length, batch_size, temperature, top_P,
                       top_K, refine_text=True, progress=gr.Progress()):
    from tts_model import generate_audio_for_seed
    from utils import split_text, replace_tokens, restore_tokens
    if seed in [0, -1, None]:
        seed = random.randint(1, 9999)
    content = ''
    if os.path.isfile(text_file):
        content = ""
    elif isinstance(text_file, str):
        content = text_file
    # Replace [uv_break] [laugh] with _uv_break_ _laugh_ and process
    content = replace_tokens(content)
    texts = split_text(content, min_length=min_length)
    for i, text in enumerate(texts):
        texts[i] = restore_tokens(text)
    print(texts)

    if oral < 0 or oral > 9 or laugh < 0 or laugh > 2 or bk < 0 or bk > 7:
        raise ValueError("oral_(0-9), laugh_(0-2), break_(0-7) out of range")

    refine_text_prompt = f"[oral_{oral}][laugh_{laugh}][break_{bk}]"
    try:
        output_files = generate_audio_for_seed(chat, seed, texts, batch_size, speed, refine_text_prompt, temperature,
                                               top_P, top_K, progress.tqdm, False, not refine_text)
        return output_files
    except Exception as e:
        return str(e)

def generate_seed():
    new_seed = random.randint(1, 9999)
    return {
        "__type__": "update",
        "value": new_seed
    }

def update_label(text):
    word_count = len(text)
    if re.search(r'\[uv_break\]|\[laugh\]', text) is not None:
        return gr.update(label=f"Reading text ({word_count} characters)>>[uv_break] [laugh] detected, it is recommended to turn off Refine to avoid unexpected audio<<")
    return gr.update(label=f"Reading text ({word_count} characters)")

def inser_token(text, btn):
    if btn == "+Laugh":
        return gr.update(
            value=text + "[laugh]"
        )
    elif btn == "+Pause":
        return gr.update(
            value=text + "[uv_break]"
        )

with gr.Blocks() as demo:
    # Project link
    gr.Markdown("""
        <div style='text-align: center; font-size: 16px;'>
            🌟  <a href='https://github.com/6drf21e/ChatTTS_colab'>Project link, welcome to star</a> 🌟
        </div>
        """)

    with gr.Tab("Voice color drawing"):
        with gr.Row():
            with gr.Column(scale=1):
                texts = [
                    "Sichuan cuisine is indeed famous for its spiciness, but there are also non-spicy options. For example, sweet water noodles, glutinous rice balls, egg cakes, and leaf cakes are all mild-flavored snacks that are sweet but not cloying, and are very popular.",
                    "I am a lively person who loves sports, travel, and trying new things. I like to challenge myself, constantly break through my own limits, and make myself stronger.",
                    "Rosen announced that it will delist on July 24th, with over 6,000 stores in China!",
                ]
                # gr.Markdown("### Random voice color drawing")
                gr.Markdown("""
                The voice color has a certain consistency under the same seed and temperature parameters. Click the "Random voice color generation" button below to generate multiple seeds. Find a satisfactory voice color and click the "Save" button below the audio.
                **Note: The voice color generated by the same seed on different machines may be different, and the voice color generated by the same seed multiple times on the same machine may also vary.**
                """)
                input_text = gr.Textbox(label="Test text",
                                        info="**Each line of text** will generate a segment of audio, and the final output audio is the result of splicing these segments together. It is recommended to use **multiple lines of text** for testing to ensure the stability of the voice color.",
                                        lines=4, placeholder="Please enter the text...", value='\n'.join(texts))

                num_seeds = gr.Slider(minimum=1, maximum=max_audio_components, step=1, label="Seed generation quantity",
                                      value=num_seeds_default)

                generate_button = gr.Button("Random voice color drawing🎲", variant="primary")

                # Saved seeds
                gr.Markdown("### Seed management interface")
                seed_list = gr.DataFrame(
                    label="Seed list",
                    headers=["Index", "Seed", "Name"],
                    datatype=["number", "number", "str"],
                    interactive=True,
                    col_count=(3, "fixed"),
                    value=display_seeds()
                )
                with gr.Row():
                    refresh_button = gr.Button("Refresh")
                    save_button = gr.Button("Save")
                    del_button = gr.Button("Delete")
                # Bind button and function
                refresh_button.click(display_seeds, outputs=seed_list)
                seed_list.select(seed_change).success(seed_change_btn, outputs=[del_button])
                save_button.click(do_save_seeds, inputs=[seed_list], outputs=None)
                del_button.click(do_delete_seed, inputs=del_button, outputs=seed_list)

            with gr.Column(scale=1):
                audio_components = []
                for i in range(max_audio_components):
                    visible = i < num_seeds_default
                    a = gr.Audio(f"Audio {i}", visible=visible)
                    t = gr.Button(f"Seed", visible=visible)
                    t.click(do_save_seed, inputs=[t], outputs=None).success(display_seeds, outputs=seed_list)
                    audio_components.append(a)
                    audio_components.append(t)

                num_seeds.change(update_audio_components, inputs=num_seeds, outputs=audio_components)

                # output = gr.Column()
                # audio = gr.Audio(label="Output Audio")

            generate_button.click(
                audio_interface_empty,
                inputs=[num_seeds, input_text],
                outputs=audio_components
            ).success(audio_interface, inputs=[num_seeds, input_text], outputs=audio_components)
    with gr.Tab("Long audio generation"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Text")
                # gr.Markdown("Please upload the text file you want to convert (.txt format).")
                # text_file_input = gr.File(label="Text file", file_types=[".txt"])
                default_text = "Sichuan cuisine is indeed famous for its spiciness, but there are also non-spicy options. For example, sweet water noodles, glutinous rice balls, egg cakes, and leaf cakes are all mild-flavored snacks that are sweet but not cloying, and are very popular."
                text_file_input = gr.Textbox(label=f"Reading text ({len(default_text)} characters)", lines=4,
                                             placeholder="Please Input Text...", value=default_text)
                # When the text box content changes, call the update_label function
                text_file_input.change(update_label, inputs=text_file_input, outputs=text_file_input)
                # Add pause button
                with gr.Row():
                    break_button = gr.Button("+Pause", variant="secondary")
                    laugh_button = gr.Button("+Laugh", variant="secondary")

            with gr.Column():
                gr.Markdown("### Configuration parameters")
                gr.Markdown("Configure the following parameters according to your needs to generate audio.")
                with gr.Row():
                    num_seeds_input = gr.Number(label="Number of audio to generate", value=1, precision=0, visible=False)
                    seed_input = gr.Number(label="Specify seed (leave blank for random)", value=None, precision=0)
                    generate_audio_seed = gr.Button("\U0001F3B2")

                with gr.Row():
                    speed_input = gr.Slider(label="Speed", minimum=1, maximum=10, value=5, step=1)
                    oral_input = gr.Slider(label="Oralization", minimum=0, maximum=9, value=2, step=1)

                    laugh_input = gr.Slider(label="Laugh", minimum=0, maximum=2, value=0, step=1)
                    bk_input = gr.Slider(label="Pause", minimum=0, maximum=7, value=4, step=1)
                # gr.Markdown("### Text parameters")
                with gr.Row():
                    # Refine
                    refine_text_input = gr.Checkbox(label="Refine",
                                                    info="When enabled, laughs and pauses will be automatically added according to the above parameters. When disabled, you can add [uv_break] [laugh] manually.",
                                                    value=True)
                    min_length_input = gr.Number(label="Text segmentation length", info="Texts longer than this value will be segmented", value=120,
                                                 precision=0)
                    batch_size_input = gr.Number(label="Batch size", info="The number of batches processed at the same time. The larger the value, the faster the speed, but it may also cause the GPU memory to explode.", value=5,
                                                 precision=0)
                with gr.Accordion("Other parameters", open=False):
                    with gr.Row():
                        # Temperature top_P top_K
                        temperature_input = gr.Slider(label="Temperature", minimum=0.01, maximum=1.0, step=0.01, value=0.3)
                        top_P_input = gr.Slider(label="top_P", minimum=0.1, maximum=0.9, step=0.05, value=0.7)
                        top_K_input = gr.Slider(label="top_K", minimum=1, maximum=20, step=1, value=20)
                        # Reset button
                        reset_button = gr.Button("Reset")

        with gr.Row():
            generate_button = gr.Button("Generate audio", variant="primary")

        with gr.Row():
            output_audio = gr.Audio(label="Generated audio file")

        generate_audio_seed.click(generate_seed,
                                  inputs=[],
                                  outputs=seed_input)

        # Reset button to reset temperature and other parameters
        reset_button.click(
            lambda: [0.3, 0.7, 20],
            inputs=None,
            outputs=[temperature_input, top_P_input, top_K_input]
        )

        generate_button.click(
            fn=generate_tts_audio,
            inputs=[
                text_file_input,
                num_seeds_input,
                seed_input,
                speed_input,
                oral_input,
                laugh_input,
                bk_input,
                min_length_input,
                batch_size_input,
                temperature_input,
                top_P_input,
                top_K_input,
                refine_text_input,
            ],
            outputs=[output_audio]
        )

        break_button.click(
            inser_token,
            inputs=[text_file_input, break_button],
            outputs=text_file_input
        )

        laugh_button.click(
            inser_token,
            inputs=[text_file_input, laugh_button],
            outputs=text_file_input
        )

    with gr.Tab("Role play"):
        def txt_2_script(text):
            lines = text.split("\n")
            data = []
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split("::")
                if len(parts) != 2:
                    continue
                data.append({
                    "character": parts[0],
                    "txt": parts[1]
                })
            return data

        def script_2_txt(data):
            assert isinstance(data, list)
            result = []
            for item in data:
                txt = item['txt'].replace('\n', ' ')
                result.append(f"{item['character']}::{txt}")
            return "\n".join(result)

        def get_characters(lines):
            assert isinstance(lines, list)
            characters = list([_["character"] for _ in lines])
            unique_characters = list(dict.fromkeys(characters))
            print([[character, 0] for character in unique_characters])
            return [[character, 0, 5, 2, 0, 4] for character in unique_characters]

        def get_txt_characters(text):
            return get_characters(txt_2_script(text))

        def llm_change(model):
            llm_setting = {
                "gpt-3.5-turbo-0125": ["https://api.openai.com/v1"],
                "gpt-4o": ["https://api.openai.com/v1"],
                "deepseek-chat": ["https://api.deepseek.com"],
                "yi-large": ["https://api.lingyiwanwu.com/v1"]
            }
            if model in llm_setting:
                return llm_setting[model][0]
            else:
                gr.Error("Model not found.")
                return None

        def ai_script_generate(model, api_base, api_key, text, progress=gr.Progress(track_tqdm=True)):
            from llm_utils import llm_operation
            from config import LLM_PROMPT
            scripts = llm_operation(api_base, api_key, model, LLM_PROMPT, text, required_keys=["txt", "character"])
            return script_2_txt(scripts)

        def generate_script_audio(text, models_seeds, progress=gr.Progress()):
            scripts = txt_2_script(text)  # Convert text to script
            characters = get_characters(scripts)  # Extract characters from the script

            #
            import pandas as pd
            from collections import defaultdict
            import itertools
            from tts_model import generate_audio_for_seed
            from utils import combine_audio, save_audio, normalize_zh
            from config import DEFAULT_BATCH_SIZE, DEFAULT_SPEED, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P

            assert isinstance(models_seeds, pd.DataFrame)

            # Batch processing function
            def batch(iterable, batch_size):
                it = iter(iterable)
                while True:
                    batch = list(itertools.islice(it, batch_size))
                    if not batch:
                        break
                    yield batch

            column_mapping = {
                'Role': 'character',
                'Seed': 'seed',
                'Speed': 'speed',
                'Oral': 'oral',
                'Laugh': 'laugh',
                'Pause': 'break'
            }
            # Rename DataFrame columns using the rename method
            models_seeds = models_seeds.rename(columns=column_mapping).to_dict(orient='records')
            # models_seeds = models_seeds.to_dict(orient='records')

            # Check if each character has a corresponding seed
            print(models_seeds)
            seed_lookup = {seed['character']: seed for seed in models_seeds}

            character_seeds = {}
            missing_seeds = []
            # Iterate over all characters
            for character in characters:
                character_name = character[0]
                seed_info = seed_lookup.get(character_name)
                if seed_info:
                    character_seeds[character_name] = seed_info
                else:
                    missing_seeds.append(character_name)

            if missing_seeds:
                missing_characters_str = ', '.join(missing_seeds)
                gr.Info(f"The following characters do not have seeds. Please set the seeds first: {missing_characters_str}")
                return None

            print(character_seeds)
            # return
            refine_text_prompt = "[oral_2][laugh_0][break_4]"
            all_wavs = []

            # Group by character to speed up inference
            grouped_lines = defaultdict(list)
            for line in scripts:
                grouped_lines[line["character"]].append(line)

            batch_results = {character: [] for character in grouped_lines}

            batch_size = 5  # Set batch size
            # Process by character
            for character, lines in progress.tqdm(grouped_lines.items(), desc="Generating script audio"):
                info = character_seeds[character]
                seed = info["seed"]
                speed = info["speed"]
                orla = info["oral"]
                laugh = info["laugh"]
                bk = info["break"]

                refine_text_prompt = f"[oral_{orla}][laugh_{laugh}][break_{bk}]"

                # Process in batches
                for batch_lines in batch(lines, batch_size):
                    texts = [normalize_zh(line["txt"]) for line in batch_lines]
                    print(f"seed={seed} t={texts} c={character} s={speed} r={refine_text_prompt}")
                    wavs = generate_audio_for_seed(chat, int(seed), texts, DEFAULT_BATCH_SIZE, speed,
                                                   refine_text_prompt, DEFAULT_TEMPERATURE, DEFAULT_TOP_P,
                                                   DEFAULT_TOP_K, skip_save=True)  # Batch processing of text
                    batch_results[character].extend(wavs)

            # Convert back to original order
            for line in scripts:
                character = line["character"]
                all_wavs.append(batch_results[character].pop(0))

            # Combine all audio
            audio = combine_audio(all_wavs)
            fname = f"script_{int(time.time())}.wav"
            return save_audio(fname, audio)

        script_example = {
            "lines": [{
                "txt": "On a sunny and beautiful afternoon, Little Red Riding Hood decided to visit her grandmother in the forest.",
                "character": "Narrator"
            }, {
                "txt": "Little Red Riding Hood said",
                "character": "Narrator"
            }, {
                "txt": "I want to bring something delicious to my grandmother.",
                "character": "Young Female"
            }, {
                "txt": "In the forest, Little Red Riding Hood encountered the cunning Big Bad Wolf.",
                "character": "Narrator"
            }, {
                "txt": "The Big Bad Wolf said",
                "character": "Narrator"
            }, {
                "txt": "Little Red Riding Hood, what do you have in your basket?",
                "character": "Middle-aged Male"
            }, {
                "txt": "Little Red Riding Hood replied",
                "character": "Narrator"
            }, {
                "txt": "These are cakes and jam for my grandmother.",
                "character": "Young Female"
            }, {
                "txt": "The Big Bad Wolf had an evil plan and decided to go to the grandmother's house first to wait for Little Red Riding Hood.",
                "character": "Narrator"
            }, {
                "txt": "When Little Red Riding Hood arrived at her grandmother's house, she found that the Big Bad Wolf had disguised himself as her grandmother.",
                "character": "Narrator"
            }, {
                "txt": "Little Red Riding Hood asked suspiciously",
                "character": "Narrator"
            }, {
                "txt": "Grandma, why are your ears so pointed?",
                "character": "Young Female"
            }, {
                "txt": "The Big Bad Wolf answered nervously",
                "character": "Narrator"
            }, {
                "txt": "Oh, it's to hear you better.",
                "character": "Middle-aged Male"
            }, {
                "txt": "Little Red Riding Hood became more and more suspicious and eventually discovered the Big Bad Wolf's trickery.",
                "character": "Narrator"
            }, {
                "txt": "She shouted for help and the hunter in the forest heard her and came to rescue her and her grandmother.",
                "character": "Narrator"
            }, {
                "txt": "Since then, Little Red Riding Hood no longer went into the forest alone and would always visit her grandmother with her family.",
                "character": "Narrator"
            }]
        }

        ai_text_default = "Wuxia novel 'Mulan vs. Zhou Tyrant' with character backgrounds"

        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                gr.Markdown("### AI Script")
                gr.Markdown("""
                To ensure stable generation results, only models equivalent to GPT-4 are supported. It is recommended to use 4o, yi-large, and deepseek.
                If there is no response, please check the error information in the log. If the format is incorrect, please try again. The models in China may be affected by the wind control, so it is recommended to change the text content and try again.

                Application channels (free quota):

                - [https://platform.deepseek.com/](https://platform.deepseek.com/)
                - [https://platform.lingyiwanwu.com/](https://platform.lingyiwanwu.com/)

                """)
                # Application channels

                with gr.Row(equal_height=True):
                    # Select model, only three options: gpt4o, deepseek-chat, yi-large
                    model_select = gr.Radio(label="Select model", choices=["gpt-4o", "deepseek-chat", "yi-large"],
                                            value="gpt-4o", interactive=True, )
                with gr.Row(equal_height=True):
                    openai_api_base_input = gr.Textbox(label="OpenAI API Base URL",
                                                       placeholder="Please enter the API Base URL",
                                                       value=r"https://api.openai.com/v1")
                    openai_api_key_input = gr.Textbox(label="OpenAI API Key", placeholder="Please enter the API Key",
                                                      value="sk-xxxxxxx")
                # AI prompt
                ai_text_input = gr.Textbox(label="Story summary or a section of the story", placeholder="Please enter the text...", lines=2,
                                           value=ai_text_default)

                # Button to generate script using AI
                ai_script_generate_button = gr.Button("AI Script Generation")

            with gr.Column(scale=3):
                gr.Markdown("### Script")
                gr.Markdown("The script can be written manually or generated using the 'AI Script Generation' button on the left. The script format is 'Role::Text', one line per sentence. Note that the role and text are separated by '::'.")
                script_text = "\n".join(
                    [f"{_.get('character', '')}::{_.get('txt', '')}" for _ in script_example['lines']])

                script_text_input = gr.Textbox(label="Script format 'Role::Text', one line per sentence. Note that the role and text are separated by '::'",
                                               placeholder="Please enter the text...",
                                               lines=12, value=script_text)
                script_translate_button = gr.Button("Step 1: Extract Roles")

            with gr.Column(scale=1):
                gr.Markdown("### Role Seeds")
                # DataFrame to store the converted script
                # Default data [speed_5][oral_2][laugh_0][break_4]
                default_data = [
                    ["Narrator", 2222, 3, 1, 0, 4],
                    ["Young Female", 2, 5, 3, 2, 4],
                    ["Middle-aged Male", 2424, 5, 2, 0, 6]
                ]

                script_data = gr.DataFrame(
                    value=default_data,
                    label="Role corresponding voice color seeds, obtained from the drawing card on the left",
                    headers=["Role", "Seed", "Speed", "Oral", "Laugh", "Pause"],
                    datatype=["str", "number", "number", "number", "number", "number"],
                    interactive=True,
                    col_count=(6, "fixed"),
                )
                # Button to generate video
                script_generate_audio = gr.Button("Step 2: Generate Audio")
        # Output of the script audio
        script_audio = gr.Audio(label="AI-generated audio", interactive=False)

        # Script-related events
        # Script conversion
        script_translate_button.click(
            get_txt_characters,
            inputs=[script_text_input],
            outputs=script_data
        )
        # Model switching event handling
        model_select.change(
            llm_change,
            inputs=[model_select],
            outputs=[openai_api_base_input]
        )
        # AI script generation
        ai_script_generate_button.click(
            ai_script_generate,
            inputs=[model_select, openai_api_base_input, openai_api_key_input, ai_text_input],
            outputs=[script_text_input]
        )
        # Audio generation
        script_generate_audio.click(
            generate_script_audio,
            inputs=[script_text_input, script_data],
            outputs=[script_audio]
        )

demo.launch(share=args.share, inbrowser=True)
