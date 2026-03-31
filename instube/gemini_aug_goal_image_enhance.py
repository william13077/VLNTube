import os
import google.generativeai as genai
import json
import time
from tqdm import tqdm
import natsort
from PIL import Image, ImageStat
import re
from instube.prompt import CAPTION_GENERATION_PROMPT,REWRITE_PROMPT_FUSION
from splits.split_utils import is_trainval



# --- API Key helper ---
def get_api_key():
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable is not set.")
        print("Please run: export GOOGLE_API_KEY='your_api_key'")
        exit()
    return api_key

# --- Helper function to extract base object name ---
def get_object_name_from_id(object_id):
    """ Extracts the base name from an object ID like 'cabinet_0001/Meshes' -> 'cabinet' """
    if not object_id or '/' not in object_id:
        return "object" # Default or fallback name
    # Use regex to find the part before the first underscore and digits
    match = re.match(r"([a-zA-Z_]+)(_\d+)?(/.*)?", object_id)
    if match:
        # Replace underscores with spaces for potentially better prompting
        return match.group(1).replace('_', ' ').strip()
    else:
        # Fallback if regex fails, try splitting
        try:
            return object_id.split('_')[0].replace('_', ' ').strip()
        except:
             return "object" # Final fallback


def get_image_path_for_goal(input_file_path, goal_key, goal_data):
    """
    Determine the image path for a goal based on the input JSON file path and goal key/data.
    """
    input_dir = os.path.dirname(input_file_path)
    image_dir = os.path.join(input_dir, 'ref', f'goal_{goal_key}.png')
    return image_dir

# --- Check if an image is potentially problematic ---
def is_image_problematic(image_path, threshold=10):
    try:
        img = Image.open(image_path).convert('L')
        stat = ImageStat.Stat(img)
        if stat.stddev[0] < threshold: return True
        if stat.mean[0] > (255 - threshold): return True
        return False
    except Exception as e:
        print(f"  - Error: Cannot process image {image_path}: {e}")
        return True

# --- Targeted caption generation ---
def generate_image_caption(image_path, model, room_id, target_name, reference_name):
    """
    Call the Gemini API to generate a targeted image caption for the goal object.
    """
    caption = "Image caption could not be generated." # Default failure message
    image_object = None

    if is_image_problematic(image_path):
        print(f"  - INFO: Image {os.path.basename(image_path)} failed quality check, skipping caption generation.")
        # Return a status reflecting the image issue
        return f"Image is problematic (e.g., black/white/blurry)."

    try:
        image_object = Image.open(image_path)
    except Exception as e:
        print(f"  - Error: Failed to load image {image_path}: {e}")
        return caption # Return failure message

    if image_object:
        try:
            # Fill the targeted prompt
            filled_caption_prompt = CAPTION_GENERATION_PROMPT.format(
                room_id=room_id if room_id else "unknown",
                target_object_name=target_name,
                reference_object_name=reference_name
            )
            prompt_parts = [filled_caption_prompt, image_object]

            response = model.generate_content(
                prompt_parts,
                 request_options={'timeout': 60}
                 )

            if hasattr(response, 'text') and response.text:
                caption = response.text.strip()
            else:
                 print(f"  - Warning: Targeted caption API did not return valid text.")
                 if hasattr(response, 'prompt_feedback'):
                    print(f"  - Prompt Feedback: {response.prompt_feedback}")
                 caption = "Targeted caption generation failed or returned empty."

        except Exception as e:
            print(f"  - Error: Exception when calling Gemini API for targeted caption: {e}")
            response_obj = locals().get('response')
            if response_obj and hasattr(response_obj, 'prompt_feedback'):
                 print(f"  - Prompt Feedback: {response_obj.prompt_feedback}")
            # Keep default failure message

    return caption


# --- Main API call function (three-step pipeline) ---
def generate_augmented_instructions(instruction_list, goal_data, room_id, model, image_path=None):
    """
    Execute three-step pipeline: 1. Extract info 2. Generate targeted caption 3. Fuse to generate instructions.
    """
    image_caption = "No image provided or caption generation skipped." # Default value

    # --- Step 1: Extract info ---
    action = "perform task" # Default action
    target_object_id = goal_data.get("object_1_id", "unknown_object")
    reference_object_id = goal_data.get("object_2_id", "unknown_reference")
    text_relation = goal_data.get("object_1_relation_to_2", "near") # Default relation

    target_name = get_object_name_from_id(target_object_id)
    reference_name = get_object_name_from_id(reference_object_id)


    # --- Step 2: Generate targeted caption ---
    if image_path:
        caption_result = generate_image_caption(image_path, model, room_id, target_name, reference_name)
        image_caption = caption_result

    # --- Step 3: Fuse text and caption to generate instructions ---
    instructions_string = "\n".join(instruction_list)
    prompt = REWRITE_PROMPT_FUSION.format(
        instructions_text=instructions_string,
        image_caption=image_caption
    )

    try:
        generation_config = {"response_mime_type": "application/json"}
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            request_options={'timeout': 120}
        )

        augmented_data = json.loads(response.text)

        if "formal" in augmented_data and "natural" in augmented_data and "casual" in augmented_data:
            usage_metadata = getattr(response, 'usage_metadata', None)
            return augmented_data, usage_metadata
        else:
            print(f"  - Error: Final instruction API returned incorrect JSON format: {response.text}")
            return None, None

    except Exception as e:
        print(f"  - Error: Exception when calling Gemini API for final instruction generation: {e}")
        response_obj = locals().get('response')
        if response_obj and hasattr(response_obj, 'prompt_feedback'):
             print(f"  - Prompt Feedback: {response_obj.prompt_feedback}")
        return None, None


# --- Main program ---
def main(input_file, output_file):
    # --- API configuration ---
    try:
        api_key = get_api_key()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"API configuration failed: {e}")
        return

    # --- Load original data ---
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} records from {input_file}")
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: {input_file} is not a valid JSON file.")
        return

    # --- Load or create output data (for checkpoint/resume) ---
    output_data = {}
    if os.path.exists(output_file):
        print(f"Found existing output file {output_file}, loading and resuming.")
        try:
            if os.path.getsize(output_file) > 0:
                 with open(output_file, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
            else:
                 print(f"  - Warning: {output_file} is empty, starting from scratch.")
                 output_data = data.copy()
        except json.JSONDecodeError:
            print(f"  - Warning: {output_file} is corrupted, starting from scratch.")
            output_data = data.copy()
        except Exception as e:
            print(f"  - Error: Failed to read {output_file}: {e}. Starting from scratch.")
            output_data = data.copy()

        if output_data and set(data.keys()) != set(output_data.keys()):
            print(f"  - Warning: {output_file} structure differs from {input_file}. Attempting to merge...")
            merged_data = data.copy()
            for key, value in merged_data.items():
                if key in output_data and isinstance(output_data.get(key), dict) and 'augmented_instructions' in output_data.get(key, {}):
                    # Make sure the instruction key exists before assigning
                    if 'instruction' not in value: value['instruction'] = {}
                    value['augmented_instructions'] = output_data[key]['augmented_instructions']
            output_data = merged_data
        elif not output_data:
            print(f"  - Warning: Failed to load {output_file}, starting from scratch.")
            output_data = data.copy()
    else:
        print(f"Output file {output_file} not found, creating new file from {input_file}.")
        output_data = data.copy()

    # --- Process and augment data ---
    save_interval = 10
    count_since_save = 0
    total_processed_in_session = 0

    print("Starting data processing...")
    keys_to_process = []
    items_already_done = 0
    total_items = len(data)

    for key, value in data.items():
        output_entry = output_data.get(key)
        is_complete = False
        if isinstance(output_entry, dict) and 'augmented_instructions' in output_entry:
             if isinstance(output_entry.get('augmented_instructions'), dict) and output_entry['augmented_instructions']:
                 is_complete = True

        if is_complete:
            items_already_done += 1
        else:
            keys_to_process.append(key)
            if key not in output_data or not isinstance(output_data.get(key), dict):
                 output_data[key] = value.copy()
            # Clean up potentially incomplete previous attempts
            elif 'augmented_instructions' in output_data.get(key, {}):
                 if isinstance(output_data.get(key), dict):
                     output_data[key].pop('augmented_instructions', None)


    print(f"Total {total_items} records.")
    print(f"Already completed {items_already_done}, will skip.")
    print(f"Need to process {len(keys_to_process)} records this run.")

    if not keys_to_process:
         print("All records have been processed.")
         return

    for key in tqdm(keys_to_process, desc="Augmenting Instructions"):
        value = output_data[key]

        original_instructions = value.get('instruction')
        goal_data = value.get('goal')
        room_id = value.get('room')
        # Validate necessary data
        if not isinstance(original_instructions, list) or not original_instructions:
            print(f"  - Warning: Key {key} has invalid or empty 'instruction' list, skipped.")
            continue
        if not isinstance(goal_data, dict) or not goal_data:
            print(f"  - Warning: Key {key} has invalid or empty 'goal' dict, skipped.")
            continue
        if not isinstance(room_id, str) or not room_id:
            print(f"  - Warning: Key {key} has invalid or empty 'room' ID, skipped.")
            continue

        image_path = get_image_path_for_goal(input_file, key, value)

        # Call the augmentation function
        augmented_data, usage = generate_augmented_instructions(
            original_instructions,
            goal_data,
            room_id,
            model,
            image_path
        )

        if usage:
             pass # Token usage logging omitted for now

        if augmented_data:
            value['augmented_instructions'] = augmented_data
            count_since_save += 1
            total_processed_in_session += 1
        else:
            print(f"  - Failed: Key {key} augmentation failed, skipped.")

        time.sleep(1.1) # Rate limiting (important!)

        if count_since_save >= save_interval and total_processed_in_session > 0:
            print(f"\n... (processed {total_processed_in_session} so far) Saving progress to {output_file} ...")
            try:
                temp_output_file = output_file + ".tmp"
                with open(temp_output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                os.replace(temp_output_file, output_file)
                count_since_save = 0
            except Exception as e:
                print(f"  - Critical error: Failed to save progress to {output_file}: {e}")

    if total_processed_in_session > 0:
        print(f"\nProcessing complete. Saving final results to {output_file} ...")
        try:
            temp_output_file = output_file + ".tmp"
            with open(temp_output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            os.replace(temp_output_file, output_file)
            print(f"All done! Processed {total_processed_in_session} records this run. Augmented data saved to {output_file}")
        except Exception as e:
            print(f"  - Critical error: Failed to save final results to {output_file}: {e}")
    elif items_already_done == total_items:
         print(f"All {total_items} records were already processed previously.")
    else:
         print(f"No new data processed this run. All pending records may have encountered errors. Please check logs.")


# --- Script entry point ---
if __name__ == "__main__":
    # --- Configure paths ---
    dataroot = '/mnt/6t/dataset/vlnverse'
    splits_file = 'splits/scene_splits.json'

    taskdir = 'goalnav_discrete'

    try:
        dirs = natsort.natsorted([p for p in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, p))])
    except FileNotFoundError:
        print(f"Error: Root directory '{dataroot}' does not exist. Please check the path.")
        exit()
    if not dirs:
        print(f"Warning: No scene directories found under '{dataroot}'.")
        exit()

    print(f"Searching for scenes under: {dataroot}")
    processed_count = 0
    for scene_id in tqdm(dirs, desc="Processing Scenes"):
        if not is_trainval(splits_file, scene_id):
            continue
        input_file = os.path.join(dataroot, scene_id, taskdir, 'goal_inst.json')
        output_file = os.path.join(dataroot, scene_id, taskdir, 'goal_inst_aug_enhance.json')

        if os.path.exists(input_file):
            print(f"\n--- Processing scene: {scene_id} ---")
            main(input_file, output_file)
            processed_count += 1
            print(f"--- Finished scene: {scene_id} ---")
        else:
            pass

    if processed_count == 0:
        print("\nNo valid 'goal_inst.json' files found to process.")
    else:
        print(f"\nFinished processing {processed_count} scenes.")
