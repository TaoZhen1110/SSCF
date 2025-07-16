import json
import requests
from tqdm import tqdm
from multiprocessing import Pool
import os


def LLM_Generated_Review(user_message, timeout=None):
    api_url = "http://172.27.33.101:8713/v1/chat/completions"
    model_path = "/mnt/data102_d2/huggingface/models/glm-4-9b-chat/"
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

    #############  Reviewer1  ###############
    prompt1 = f"""
    Assume you are a top-tier conference reviewer. Please read the entire paper and write a single-paragraph review.
    
    Here is the paper:

    {Paper_Text}
    
    Finally, you only need to output your review texts.
    """

    LLM_Review1 = LLM_Generated_Review(prompt1).replace('\n', ' ').replace('  ', ' ')
    review_item = {
        "Paper_ID": data["Paper_ID"]
    }

    review_item["LLM_Review1"] = LLM_Review1


    #############  Reviewer2  ###############
    prompt2 = f"""
    You are serving as a reviewer for a top-tier conference.\ 
    Please carefully read the entire paper, evaluate its quality, and write a single-paragraph review.\ 
    Your review should include an assessment of the paper's originality, technical rigor, clarity of presentation,\ 
    and relevance to the field. Additionally, highlight the paper's strengths,\ 
    note any significant weaknesses or areas for improvement, and provide constructive feedback to the authors.
    
    Here is the paper:
    
    {Paper_Text}
    
    Finally, you only need to output your review texts.
    """

    LLM_Review2 = LLM_Generated_Review(prompt2).replace('\n', ' ').replace('  ', ' ')
    review_item["LLM_Review2"] = LLM_Review2

    #############  Reviewer3  ###############
    prompt3 = f"""
    You are an academic reviewer. Write a single-paragraph review for the following research paper. Include:
    1. A brief summary of the paper.
    2. Strengths of the study.
    3. Weaknesses or limitations.
    4. Suggestions for improvement.

    Here is the paper:
    
    {Paper_Text}
    
    Finally, you only need to output your review texts.
    """

    LLM_Review3 = LLM_Generated_Review(prompt3).replace('\n', ' ').replace('  ', ' ')
    review_item["LLM_Review3"] = LLM_Review3

    #############  Reviewer4  ###############
    prompt4 = f"""
    You are an experienced academic reviewer for a peer-reviewed conference.\
    Your task is to review the following English research paper critically and constructively.\ 
    Your output should be like the following format: 

    Summary:  
    Strengths: 
    Weaknesses:  
    Suggestions:


    Here is the paper:
    
    {Paper_Text}
    
    Finally, you only need to output your review texts.
    """

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
                # 达到最大尝试次数，可以选择返回一个特定的错误标记，或者抛出异常
                return None  # 或者 raise


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

    Paper_file = "/mnt/data132/taozhen/LLM_Review_Detection/Dataset/Human_Review/ACL/papers.json"
    LLM_review_file = "/mnt/data132/taozhen/LLM_Review_Detection/Dataset/LLM_Review/GLM/ACL1.json"

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





