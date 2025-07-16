import json
import requests
from tqdm import tqdm
from multiprocessing import Pool
import os


def truncate_paper_text(paper_text, prompt_template, max_model_len=32766):
    """
    按字符长度截断 Paper_Text，确保总长度不会超过 max_model_len。
    """
    prompt_length = len(prompt_template)
    max_paper_length = max_model_len - prompt_length

    if len(paper_text) > max_paper_length:
        return paper_text[:max_paper_length]
    return paper_text


def LLM_Generated_Review(user_message, timeout=None):
    """
    调用 API 并返回生成结果。
    """
    api_url = "http://172.27.33.102:8711/v1/chat/completions"
    model_path = "/mnt/data102_d2/huggingface/models/Qwen2.5-32B-Instruct/"
    temperature = 0.9

    payload = {
        "model": model_path,
        "messages": [
            {
                "role": "user",
                "content": user_message
            },
        ],
        "temperature": temperature,
    }

    try:
        response = requests.post(api_url, data=json.dumps(payload), timeout=timeout)
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        response_data = response.json()

        if 'choices' in response_data and len(response_data['choices']) > 0:
            return response_data['choices'][0]['message']['content']
        else:
            return "No story generated. Check the API response for more details."
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"
    except json.JSONDecodeError:
        return "Failed to decode the JSON response from the API."
    except KeyError as e:
        return f"Missing key in the API response: {e}"


def process_data(data):
    Paper_Text = data["Paper_Text"]
    review_item = {"Paper_ID": data["Paper_ID"]}

    #############  Reviewer1  ###############
    prompt1_template = """
    Assume you are a top-tier conference reviewer. Please read the entire paper and write a single-paragraph review.

    Here is the paper:
    """
    prompt1 = f"{prompt1_template}\n\n{Paper_Text}\n\nFinally, you only need to output your review texts."

    LLM_Review1 = LLM_Generated_Review(prompt1).replace('\n', ' ').replace('  ', ' ')
    if "An error occurred" in LLM_Review1:
        print(f"Error for LLM_Review1, retrying with truncated text...")
        truncated_text = truncate_paper_text(Paper_Text, prompt1_template, max_model_len=32766)
        prompt1 = f"{prompt1_template}\n\n{truncated_text}\n\nFinally, you only need to output your review texts."
        LLM_Review1 = LLM_Generated_Review(prompt1).replace('\n', ' ').replace('  ', ' ')
    review_item["LLM_Review1"] = LLM_Review1

    #############  Reviewer2  ###############
    prompt2_template = """
    You are serving as a reviewer for a top-tier conference. 
    Please carefully read the entire paper, evaluate its quality, and write a single-paragraph review.
    Your review should include an assessment of the paper's originality, technical rigor, clarity of presentation,
    and relevance to the field. Additionally, highlight the paper's strengths,
    note any significant weaknesses or areas for improvement, and provide constructive feedback to the authors.

    Here is the paper:
    """
    prompt2 = f"{prompt2_template}\n\n{Paper_Text}\n\nFinally, you only need to output your review texts."

    LLM_Review2 = LLM_Generated_Review(prompt2).replace('\n', ' ').replace('  ', ' ')
    if "An error occurred" in LLM_Review2:
        print(f"Error for LLM_Review2, retrying with truncated text...")
        truncated_text = truncate_paper_text(Paper_Text, prompt2_template, max_model_len=32766)
        prompt2 = f"{prompt2_template}\n\n{truncated_text}\n\nFinally, you only need to output your review texts."
        LLM_Review2 = LLM_Generated_Review(prompt2).replace('\n', ' ').replace('  ', ' ')
    review_item["LLM_Review2"] = LLM_Review2

    #############  Reviewer3  ###############
    prompt3_template = """
    You are an academic reviewer. Write a single-paragraph review for the following research paper. Include:
    1. A brief summary of the paper.
    2. Strengths of the study.
    3. Weaknesses or limitations.
    4. Suggestions for improvement.

    Here is the paper:
    """
    prompt3 = f"{prompt3_template}\n\n{Paper_Text}\n\nFinally, you only need to output your review texts."

    LLM_Review3 = LLM_Generated_Review(prompt3).replace('\n', ' ').replace('  ', ' ')
    if "An error occurred" in LLM_Review3:
        print(f"Error for LLM_Review3, retrying with truncated text...")
        truncated_text = truncate_paper_text(Paper_Text, prompt3_template, max_model_len=32766)
        prompt3 = f"{prompt3_template}\n\n{truncated_text}\n\nFinally, you only need to output your review texts."
        LLM_Review3 = LLM_Generated_Review(prompt3).replace('\n', ' ').replace('  ', ' ')
    review_item["LLM_Review3"] = LLM_Review3

    #############  Reviewer4  ###############
    prompt4_template = """
    You are an experienced academic reviewer for a peer-reviewed conference.
    Your task is to review the following English research paper critically and constructively.
    Your output should be like the following format: 

    Summary:  
    Strengths: 
    Weaknesses:  
    Suggestions:

    Here is the paper:
    """
    prompt4 = f"{prompt4_template}\n\n{Paper_Text}\n\nFinally, you only need to output your review texts."

    LLM_Review4 = LLM_Generated_Review(prompt4).replace('\n', ' ').replace('  ', ' ')
    if "An error occurred" in LLM_Review4:
        print(f"Error for LLM_Review4, retrying with truncated text...")
        truncated_text = truncate_paper_text(Paper_Text, prompt4_template, max_model_len=32766)
        prompt4 = f"{prompt4_template}\n\n{truncated_text}\n\nFinally, you only need to output your review texts."
        LLM_Review4 = LLM_Generated_Review(prompt4).replace('\n', ' ').replace('  ', ' ')
    review_item["LLM_Review4"] = LLM_Review4

    return review_item


def process_line_with_retry(line, max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        try:
            return process_data(line)
        except Exception as e:
            print(f"处理失败，尝试次数 {attempt}/{max_attempts}: {e}")
            if attempt == max_attempts:
                return None


def save_data(data, file_path):
    """
    将处理后的数据立即保存到文件中。
    """
    with open(file_path, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":

    processes = 10
    p = Pool(processes=processes)

    Paper_file = "/mnt/data132/taozhen/LLM_Review_Detection/Dataset/Human_Review/ACL/Papers.json"
    LLM_review_file = "/mnt/data132/taozhen/LLM_Review_Detection/Dataset/LLM_Review/Qwen/ACL1.json"

    # 如果输出文件不存在，创建并处理数据
    if not os.path.exists(LLM_review_file):
        print(f"Creating {LLM_review_file}")

        # 读取输入数据
        with open(Paper_file, 'r', encoding='utf-8') as file:
            data_list = [item for item in json.load(file)]

        # 使用 imap_unordered 获取迭代器，允许在任务完成时立即处理结果
        with tqdm(total=len(data_list), desc="Processing Data") as progress_bar:
            for result in p.imap_unordered(process_line_with_retry, data_list):
                if result is not None:
                    save_data(result, LLM_review_file)
                    progress_bar.update(1)  # 更新进度条

    else:
        print(f"Loading {LLM_review_file} to check missing items")
        existing_ids = set()
        missing_items = []

        # 读取已有的文件，并记录已有的 ID
        with open(LLM_review_file, 'r', encoding='utf-8') as existing_file:
            for line in existing_file:
                data = json.loads(line)
                existing_ids.add(data['Paper_ID'])

        # 读取源数据文件，查找缺失的条目
        with open(Paper_file, 'r', encoding='utf-8') as original_file:
            data_list = [item for item in json.load(original_file)]

        # 识别缺失项
        for item in data_list:
            if item['Paper_ID'] not in existing_ids:
                missing_items.append(item)

        # 处理缺失的条目
        with tqdm(total=len(missing_items), desc="Processing Missing Items") as progress_bar:
            for result in p.imap_unordered(process_line_with_retry, missing_items):
                if result is not None:
                    save_data(result, LLM_review_file)
                    progress_bar.update(1)

    p.close()
    p.join()
