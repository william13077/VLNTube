import os
import time
import google.generativeai as genai
import natsort
import json
from PIL import Image
from tqdm import tqdm
from instube.prompt import VLN_PROMPT_IMAGE_SEQUENCE

# --- Max frames to send per API call ---
MAX_FRAMES_TO_SEND = 30

def get_api_key():
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable is not set.")
        exit()
    return api_key


def analyze_image_sequence_with_prompt(image_paths: list, user_prompt: str):
    """
    Analyze an image sequence using the Gemini API.

    Args:
        image_paths (list): List of local image file paths in chronological order.
        user_prompt (str): The instruction prompt text.

    Returns:
        tuple: (generated text response, usage metadata)
    """
    try:
        api_key = get_api_key()
        genai.configure(api_key=api_key)
    except Exception as e:
        return f"API configuration failed: {e}"

    print(f"Preparing to process {len(image_paths)} sequence images..")

    # Prepare model (gemini-2.5-flash supports large image inputs)
    # model = genai.GenerativeModel(model_name="models/gemini-2.5-pro")
    model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")


    try:
        # Build prompt list (Pillow Image objects are handled automatically by the SDK)
        prompt_parts = [user_prompt]


        # Append image sequence
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f" {img_path} not exist, skip...")
                continue
            prompt_parts.append(Image.open(img_path))

        print("Ready, sending request to Gemini...")
        response = model.generate_content(prompt_parts)

        # Return result
        return response.text, response.usage_metadata

    except Exception as e:
        return f"Error calling Gemini API: {e}"

# --- Main entry point ---
if __name__ == "__main__":
    dataroot = '/mnt/6t/dataset/vlnverse'
    dir = natsort.natsorted([i for i in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, i))])


    task_dir = 'goalnav_discrete'
    seq_dir = 'sequence_discrete'
    inst_dir = 'inst'
    json_name = 'inst_img_sequence.json'

    for scene_id in dir:

        save_dir = os.path.join(dataroot, scene_id, task_dir, inst_dir)
        output_file = os.path.join(save_dir, json_name)
        os.makedirs(save_dir, exist_ok=True)
        results = {}

        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"Loaded: {output_file}, containing {len(results)} records.")
            except json.JSONDecodeError:
                results = {}
        sequence_folders = natsort.natsorted( [ i for i in os.listdir(os.path.join(dataroot, scene_id, task_dir, seq_dir)) if os.path.isdir(os.path.join(dataroot, scene_id, task_dir, seq_dir, i)) ])


        for seq_folder_name in tqdm(sequence_folders, desc=f"Processing sequences in scene {scene_id}"):
            if seq_folder_name in results:
                continue

            current_seq_path = os.path.join(dataroot, scene_id, task_dir, seq_dir, seq_folder_name)

            all_image_paths = natsort.natsorted(
                [os.path.join(current_seq_path, f) for f in os.listdir(current_seq_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.lower().startswith('rgb_')]
            )
            if not all_image_paths:
                print(f"No images found in sequence folder {current_seq_path}, skipped.")
                continue

            num_images = len(all_image_paths)
            if num_images > MAX_FRAMES_TO_SEND:
                # Too many images, uniformly sample
                indices = [int(i * (num_images - 1) / (MAX_FRAMES_TO_SEND - 1)) for i in range(MAX_FRAMES_TO_SEND)]
                sampled_image_paths = [all_image_paths[i] for i in sorted(list(set(indices)))] # deduplicate
            else:
                # Within limit, use all images
                sampled_image_paths = all_image_paths

            try:
                print(f"Processing sequence: {seq_folder_name}. Total {num_images} images, sampled {len(sampled_image_paths)}.")

                result = analyze_image_sequence_with_prompt(
                    sampled_image_paths,
                    VLN_PROMPT_IMAGE_SEQUENCE
                )

                results[seq_folder_name] = {
                    "sequence_path": current_seq_path,
                    "instruction": result[0],
                    "num_original_frames": num_images,
                    "num_sampled_frames": len(sampled_image_paths)
                }
                print(f"Result: {result}")
                print("\n--------------------------\n")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"\nError processing sequence {seq_folder_name}: {e}")
                print("Skipping and continuing...")
                continue

        print(f"All results saved to: {output_file}")
