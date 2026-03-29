import os
import time
import google.generativeai as genai
import natsort
import json
from PIL import Image 
from tqdm import tqdm
from prompt import VLN_PROMPT_IMAGE_SEQUENCE

# --- 新增：定义一次API调用中发送的最大帧数 ---
MAX_FRAMES_TO_SEND = 30

def get_api_key():
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("错误：GOOGLE_API_KEY 环境变量未设置。")
        exit()
    return api_key


def analyze_image_sequence_with_prompt(image_paths: list, user_prompt: str):
    """
    使用 Gemini API 分析一个图像序列和一张BEV图像。

    参数:
    - image_paths (list): 按顺序排列的本地图像文件路径列表。
    - user_prompt (str): 你的指令文本。

    返回:
    - str: 模型生成的文本响应。
    """
    try:
        api_key = get_api_key()
        genai.configure(api_key=api_key)
    except Exception as e:
        return f"API 配置失败: {e}"

    print(f"准备处理 {len(image_paths)} 张序列图片..")

    # 3. 准备调用模型
    # gemini-1.5-flash 或 gemini-1.5-pro 都支持大量的图像输入
    # model = genai.GenerativeModel(model_name="models/gemini-2.5-pro")
    model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")


    try:
        # 4. 构建 Prompt 列表
        # 直接传递Pillow Image对象，SDK会自动处理
        prompt_parts = [user_prompt]
        

        # 添加图像序列
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f" {img_path} not exist, skip...")
                continue
            prompt_parts.append(Image.open(img_path))
        
        print("准备就绪，正在向 Gemini 发送请求...")
        response = model.generate_content(prompt_parts)
        
        # 5. 返回结果
        # return response.text
        return response.text, response.usage_metadata
        
    except Exception as e:
        return f"调用 Gemini API 时发生错误: {e}"

# --- 主程序入口 ---
if __name__ == "__main__":
    dataroot = '/mnt/6t/dataset/vlnverse'
    dir = natsort.natsorted([i for i in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, i))])


    task_dir = 'goalnav_discrete'
    seq_dir = 'sequence_discrete'
    inst_dir = 'inst'
    json_name = 'inst_img_sequence.json' # 修改输出文件名

    for scene_id in dir: 

        save_dir = os.path.join(dataroot, scene_id, task_dir, inst_dir)
        output_file = os.path.join(save_dir, json_name)
        os.makedirs(save_dir, exist_ok=True)
        results = {}

        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"已加载: {output_file}，包含 {len(results)} 条记录。")
            except json.JSONDecodeError:
                results = {}
        sequence_folders = natsort.natsorted( [ i for i in os.listdir(os.path.join(dataroot, scene_id, task_dir, seq_dir)) if os.path.isdir(os.path.join(dataroot, scene_id, task_dir, seq_dir, i)) ])

        
        for seq_folder_name in tqdm(sequence_folders, desc=f"处理场景 {scene_id} 中的序列"):
            if seq_folder_name in results:
                continue
            
            current_seq_path = os.path.join(dataroot, scene_id, task_dir, seq_dir, seq_folder_name)
 
            all_image_paths = natsort.natsorted(
                [os.path.join(current_seq_path, f) for f in os.listdir(current_seq_path) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and f.lower().startswith('rgb_')]
            )
            if not all_image_paths:
                print(f"序列文件夹 {current_seq_path} 中没有找到图片，已跳过。")
                continue
            
            num_images = len(all_image_paths)
            if num_images > MAX_FRAMES_TO_SEND:
                # 如果图片太多，进行均匀采样
                indices = [int(i * (num_images - 1) / (MAX_FRAMES_TO_SEND - 1)) for i in range(MAX_FRAMES_TO_SEND)]
                sampled_image_paths = [all_image_paths[i] for i in sorted(list(set(indices)))] # set去重
            else:
                # 如果图片数量在限制内，全部使用
                sampled_image_paths = all_image_paths

            try:
                print(f"处理序列: {seq_folder_name}。共 {num_images} 张图片，采样 {len(sampled_image_paths)} 张。")
                
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
                print(f"结果: {result}")
                print("\n--------------------------\n")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"\n处理序列 {seq_folder_name} 时发生错误: {e}")
                print("跳过并继续...")
                continue
                
        print(f"所有结果已保存至: {output_file}")