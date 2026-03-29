import os
import google.generativeai as genai
import json
import time
from tqdm import tqdm
import natsort
from PIL import Image, ImageStat # 导入Pillow库用于图像处理
import re # 导入正则表达式库用于提取对象名称
from prompt import CAPTION_GENERATION_PROMPT,REWRITE_PROMPT_FUSION



# --- 2. API Key 函数 (保持不变) ---
def get_api_key():
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("错误：GOOGLE_API_KEY 环境变量未设置。")
        print("请在终端运行: export GOOGLE_API_KEY='你的API_KEY'")
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
    根据输入json文件路径和goal的key/数据，确定对应的图片路径。
    *** 你需要根据你的实际文件结构修改此函数的逻辑 ***
    """
    # 示例逻辑：假设图片在json文件同目录下的 'goal_images' 子目录中，
    # 并且图片文件名是 goal key 加上 .jpg 后缀。
    # 例如： goal_key = "0" -> <input_dir>/goal_images/0.jpg
    #       goal_key = "path_0_1" -> <input_dir>/goal_images/path_0_1.jpg
    
    input_dir = os.path.dirname(input_file_path)
    # breakpoint()
    # --- 你可能需要修改下面的路径 ---
    # image_dir = os.path.join(input_dir, '..', 'sequence_1', goal_key) # 尝试从之前的sequence路径推断
    image_dir = os.path.join(input_dir, 'ref', f'goal_{goal_key}.png') 
    return image_dir

# --- 更新: 检查图像是否可能有问题 (保持不变) ---
def is_image_problematic(image_path, threshold=10):
    try:
        img = Image.open(image_path).convert('L')
        stat = ImageStat.Stat(img)
        if stat.stddev[0] < threshold: return True
        if stat.mean[0] > (255 - threshold): return True
        return False
    except Exception as e:
        print(f"  - 错误: 无法处理图片 {image_path}: {e}")
        return True

# --- 更新: Targeted Caption 生成函数 ---
def generate_image_caption(image_path, model, room_id, target_name, reference_name):
    """
    调用 Gemini API 生成针对目标物体的图像 caption。
    """
    caption = "Image caption could not be generated." # Default failure message
    image_object = None

    if is_image_problematic(image_path):
        print(f"  - INFO: 图片 {os.path.basename(image_path)} 质量检查未通过，不生成Caption。")
        # Return a status reflecting the image issue
        return f"Image is problematic (e.g., black/white/blurry)."

    try:
        image_object = Image.open(image_path)
    except Exception as e:
        print(f"  - 错误: 加载图片失败 {image_path}: {e}")
        return caption # Return failure message

    if image_object:
        try:
            # print(f"  - INFO: 为图片 {os.path.basename(image_path)} 生成 Targeted Caption...")
            # Fill the targeted prompt
            filled_caption_prompt = CAPTION_GENERATION_PROMPT.format(
                room_id=room_id if room_id else "unknown",
                target_object_name=target_name,
                reference_object_name=reference_name
            )
            prompt_parts = [filled_caption_prompt, image_object]

            response = model.generate_content(
                prompt_parts,
                 request_options={'timeout': 60} # Captioning is usually fast
                 )

            if hasattr(response, 'text') and response.text:
                caption = response.text.strip()
                # print(f"  - INFO: 生成的 Targeted Caption: '{caption}'")
            else:
                 print(f"  - 警告: Targeted Caption 生成 API 未返回有效文本。")
                 if hasattr(response, 'prompt_feedback'):
                    print(f"  - Prompt Feedback: {response.prompt_feedback}")
                 caption = "Targeted caption generation failed or returned empty."

        except Exception as e:
            print(f"  - 错误: 调用 Gemini API 生成 Targeted Caption 时发生异常: {e}")
            response_obj = locals().get('response')
            if response_obj and hasattr(response_obj, 'prompt_feedback'):
                 print(f"  - Prompt Feedback: {response_obj.prompt_feedback}")
            # Keep default failure message

    return caption


# --- 3. 修改后的主 API 调用函数 (三步流程) ---
def generate_augmented_instructions(instruction_list, goal_data, room_id, model, image_path=None):
    """
    执行三步流程：1. 提取信息 2. 生成Targeted Caption 3. 融合生成指令。
    """
    image_caption = "No image provided or caption generation skipped." # Default value

    # --- 第一步: 提取信息 ---
    action = "perform task" # Default action
    target_object_id = goal_data.get("object_1_id", "unknown_object")
    reference_object_id = goal_data.get("object_2_id", "unknown_reference")
    text_relation = goal_data.get("object_1_relation_to_2", "near") # Default relation

    target_name = get_object_name_from_id(target_object_id)
    reference_name = get_object_name_from_id(reference_object_id)


    # --- 第二步: 生成 Targeted Caption ---
    if image_path:
        # Pass necessary info to caption generation
        caption_result = generate_image_caption(image_path, model, room_id, target_name, reference_name)
        image_caption = caption_result # Use the result, even if it indicates failure/problem

    # --- 第三步: 融合文本和 Caption 生成指令 ---
    instructions_string = "\n".join(instruction_list)
    prompt = REWRITE_PROMPT_FUSION.format(
        instructions_text=instructions_string,
        image_caption=image_caption
        # Note: We don't need to pass action/objects here, as the fusion prompt
        # asks the model to derive the core goal from the 10 text instructions.
        # The caption's role is primarily for spatial relationship and details.
    )

    try:
        generation_config = {"response_mime_type": "application/json"}
        # print(f"  - INFO: 正在融合文本和 Targeted Caption ('{image_caption[:50]}...') 生成指令...")
        response = model.generate_content(
            prompt, # Third step only needs text input
            generation_config=generation_config,
            request_options={'timeout': 120}
        )

        augmented_data = json.loads(response.text)

        if "formal" in augmented_data and "natural" in augmented_data and "casual" in augmented_data:
            usage_metadata = getattr(response, 'usage_metadata', None)
            return augmented_data, usage_metadata
        else:
            print(f"  - 错误: 最终指令生成 API 返回的JSON格式不正确: {response.text}")
            return None, None

    except Exception as e:
        print(f"  - 错误: 调用 Gemini API 进行最终指令生成时发生异常: {e}")
        response_obj = locals().get('response')
        if response_obj and hasattr(response_obj, 'prompt_feedback'):
             print(f"  - Prompt Feedback: {response_obj.prompt_feedback}")
        return None, None


# --- 4. 主程序 (调用逻辑更新) ---
def main(input_file, output_file):
    # --- API 配置 ---
    try:
        api_key = get_api_key()
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")
        print("Gemini API 配置成功。")
    except Exception as e:
        print(f"API 配置失败: {e}")
        return

    # --- 加载原始数据 ---
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载 {len(data)} 条记录 from {input_file}")
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")
        return
    except json.JSONDecodeError:
        print(f"错误: {input_file} 不是一个有效的JSON文件。")
        return

    # --- 加载或创建输出数据（用于断点续传） ---
    output_data = {}
    if os.path.exists(output_file):
        print(f"检测到已存在的输出文件 {output_file}，将加载并继续。")
        try:
            if os.path.getsize(output_file) > 0:
                 with open(output_file, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
            else:
                 print(f"  - 警告: {output_file} 为空，将从头开始。")
                 output_data = data.copy()
        except json.JSONDecodeError:
            print(f"  - 警告: {output_file} 损坏，将从头开始。")
            output_data = data.copy()
        except Exception as e:
            print(f"  - 错误: 读取 {output_file} 时发生错误: {e}. 将从头开始。")
            output_data = data.copy()

        if output_data and set(data.keys()) != set(output_data.keys()):
            print(f"  - 警告: {output_file} 的结构与 {input_file} 不一致。尝试合并...")
            merged_data = data.copy()
            for key, value in merged_data.items():
                if key in output_data and isinstance(output_data.get(key), dict) and 'augmented_instructions' in output_data.get(key, {}):
                    # Make sure the instruction key exists before assigning
                    if 'instruction' not in value: value['instruction'] = {}
                    value['augmented_instructions'] = output_data[key]['augmented_instructions']
            output_data = merged_data
        elif not output_data:
            print(f"  - 警告: 加载 {output_file} 失败，将从头开始。")
            output_data = data.copy()
    else:
        print(f"未找到输出文件 {output_file}，将从 {input_file} 创建新文件。")
        output_data = data.copy()

    # --- 循环处理并增强数据 ---
    save_interval = 10
    count_since_save = 0
    total_processed_in_session = 0

    print("开始处理数据...")
    keys_to_process = []
    items_already_done = 0
    total_items = len(data)

    for key, value in data.items():
        output_entry = output_data.get(key)
        is_complete = False
        # Adjusted check to look inside value, which now exists in output_data
        if isinstance(output_entry, dict) and 'augmented_instructions' in output_entry:
             if isinstance(output_entry.get('augmented_instructions'), dict) and output_entry['augmented_instructions']:
                 is_complete = True

        if is_complete:
            items_already_done += 1
        else:
            keys_to_process.append(key)
            if key not in output_data or not isinstance(output_data.get(key), dict):
                 output_data[key] = value.copy()
            # Clean up potentially incomplete previous attempts more reliably
            elif 'augmented_instructions' in output_data.get(key, {}):
                 # Ensure the key exists and is a dict before trying to delete
                 if isinstance(output_data.get(key), dict):
                     output_data[key].pop('augmented_instructions', None)


    print(f"总共 {total_items} 条记录。")
    print(f"已完成 {items_already_done} 条，本次将跳过。")
    print(f"本次需处理 {len(keys_to_process)} 条。")

    if not keys_to_process:
         print("所有记录均已处理完毕。")
         return

    for key in tqdm(keys_to_process, desc="Augmenting Instructions"):
        value = output_data[key]

        original_instructions = value.get('instruction')
        goal_data = value.get('goal')
        room_id = value.get('room') # Now directly getting room_id
        # Validate necessary data
        if not isinstance(original_instructions, list) or not original_instructions:
            print(f"  - 警告: Key {key} 的 'instruction' 列表无效或为空，已跳过。")
            continue
        if not isinstance(goal_data, dict) or not goal_data:
            print(f"  - 警告: Key {key} 的 'goal' 字典无效或为空，已跳过。")
            continue
        if not isinstance(room_id, str) or not room_id:
            print(f"  - 警告: Key {key} 的 'room' ID 无效或为空，已跳过。")
            continue

        image_path = get_image_path_for_goal(input_file, key, value)

        # 调用更新后的主函数，传入必要信息
        augmented_data, usage = generate_augmented_instructions(
            original_instructions,
            goal_data,
            room_id,
            model,
            image_path
        )

        if usage:
             pass # 暂时不打印 token 使用量

        if augmented_data:
            value['augmented_instructions'] = augmented_data
            count_since_save += 1
            total_processed_in_session += 1
        else:
            print(f"  - 失败: Key {key} 增强失败，已跳过。")

        time.sleep(1.1) # 保持速率限制 (重要!)

        if count_since_save >= save_interval and total_processed_in_session > 0:
            print(f"\n... (本次已处理 {total_processed_in_session} 条) 正在保存进度到 {output_file} ...")
            try:
                temp_output_file = output_file + ".tmp"
                with open(temp_output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                os.replace(temp_output_file, output_file)
                count_since_save = 0
            except Exception as e:
                print(f"  - 严重错误: 保存进度到 {output_file} 时失败: {e}")

    if total_processed_in_session > 0:
        print(f"\n处理完成。正在保存最终结果到 {output_file} ...")
        try:
            temp_output_file = output_file + ".tmp"
            with open(temp_output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            os.replace(temp_output_file, output_file)
            print(f"全部完成！本次运行共处理 {total_processed_in_session} 条记录。增强后的数据已保存至 {output_file}")
        except Exception as e:
            print(f"  - 严重错误: 保存最终结果到 {output_file} 时失败: {e}")
    elif items_already_done == total_items:
         print(f"所有 {total_items} 条记录先前已处理完毕。")
    else:
         print(f"本次运行未处理新数据，可能所有待处理数据都遇到了错误。请检查日志。")


# --- 脚本入口 ---
if __name__ == "__main__":
    # --- 配置你的路径 ---
    dataroot = '/mnt/6t/dataset/cvpr_kl'

    taskdir = 'goalnav_discrete'

    try:
        dirs = natsort.natsorted([p for p in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, p))])
    except FileNotFoundError:
        print(f"错误: 根目录 '{dataroot}' 不存在。请检查路径。")
        exit()
    if not dirs:
        print(f"警告: 在 '{dataroot}' 下未找到任何场景目录。")
        exit()

    print(f"将在以下根目录下查找场景: {dataroot}")
    processed_count = 0
    for scene_id in tqdm(dirs, desc="Processing Scenes"):
        input_file = os.path.join(dataroot, scene_id, taskdir, 'goal_inst.json')
        output_file = os.path.join(dataroot, scene_id, taskdir, 'goal_inst_aug_enhance.json')

        if os.path.exists(input_file):
            print(f"\n--- 开始处理场景: {scene_id} ---")
            main(input_file, output_file)
            processed_count += 1
            print(f"--- 完成处理场景: {scene_id} ---")
        else:
            pass

    if processed_count == 0:
        print("\n未找到任何有效的 'goal_inst.json' 文件进行处理。")
    else:
        print(f"\n已完成对 {processed_count} 个场景的处理。")

