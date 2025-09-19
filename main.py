import os
import re
import json
import arxiv
import yaml
import logging
import argparse
import datetime
import requests
from typing import Optional
import time
import google.generativeai as genai


# https://github.com/google-gemini/cookbook/tree/main
# https://ai.google.dev/api?hl=zh-cn
class Translater:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)  # 填入自己的 API Key

        # 查询模型
        for m in genai.list_models():
            print(m.name)
            print(m.supported_generation_methods)
        sys_prompt = (
            "You are a highly skilled translator specializing in artificial intelligence and computer science. "
            "You pride yourself on incredible accuracy and attention to detail. You always stick to the facts in the sources provided, and never make up new facts. "
            "Your translations are known for their accuracy, clarity, and fluency.\n"
            "Your task is to translate technical academic abstracts from English to Simplified Chinese. "
            "You will receive an English abstract, and you should produce a Chinese translation that adheres to the following:\n"
            "* **Accuracy:** All technical terms and concepts must be translated correctly.\n"
            "* **Clarity:** The translation should be easily understood by someone familiar with AI concepts.\n"
            "* **Fluency:** The translation should read naturally in Chinese.\n"
            "* **Output Format:** The returned text should not be bolded, not be separated into paragraphs, and remove all line breaks to merge into a single paragraph.\n"
            "Do not add your own opinions or interpretations; remain faithful to the original text while optimizing for readability. "
        )

        self.model = genai.GenerativeModel(
            "gemini-2.5-flash",
            system_instruction=sys_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.8,
            ),
        )

    def translate(self, text: str):
        response = self.model.generate_content(
            f"Note output format, here is the abstract to translate:\n{text}"
        )
        return response.text


logging.basicConfig(
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

github_url = "https://api.github.com/search/repositories"
arxiv_url = "http://arxiv.org/"


def load_config(config_file: str) -> dict:
    """加载配置文件"""

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        logging.info(f"config = {config}")
    return config


def get_authors(authors, first_author=False):
    """获取作者名（支持仅取第一作者）"""
    if first_author:
        return authors[0] if authors else ""
    return ", ".join(str(author) for author in authors)


def sort_papers(papers):
    """按论文 ID 降序排列论文"""
    output = {}
    keys = sorted(papers.keys(), reverse=True)
    for key in keys:
        output[key] = papers[key]
    return output


def get_code_link(qword: str) -> str:
    """从 GitHub 搜索论文相关代码仓库，添加错误处理"""
    try:
        params = {"q": qword, "sort": "stars", "order": "desc"}
        # 添加超时设置，避免无限等待
        response = requests.get(github_url, params=params, timeout=10)
        
        # 检查请求是否成功（状态码200）
        if response.status_code != 200:
            logging.warning(f"GitHub API请求失败，状态码: {response.status_code}，查询词: {qword}")
            return None
            
        results = response.json()
        
        # 检查是否包含'total_count'字段
        if "total_count" not in results:
            logging.warning(f"GitHub API响应格式异常，缺少total_count字段: {results}")
            return None
            
        return results["items"][0]["html_url"] if results["total_count"] > 0 else None
        
    except requests.exceptions.Timeout:
        logging.error(f"GitHub API请求超时，查询词: {qword}")
        return None
    except requests.exceptions.RequestException as e:
        logging.error(f"GitHub API请求异常: {str(e)}，查询词: {qword}")
        return None
    except (KeyError, IndexError) as e:
        logging.error(f"解析GitHub API响应失败: {str(e)}，响应内容: {results}")
        return None


def get_daily_papers(
    topic, query="slam", max_results=2, translater: Optional[Translater] = None
):
    """获取每日论文（含翻译、代码链接匹配）"""
    content = {}
    content_to_web = {}
    print(f"query = {query}")
    search_engine = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate
    )

    for result in search_engine.results():
        paper_id = result.get_short_id()
        paper_title = result.title
        paper_url = arxiv_url + "abs/" + paper_id.split("v")[0]  # 移除版本号
        paper_abstract = result.summary.replace("\n", " ").rstrip()
        update_time = result.updated.date()

        # 翻译摘要（若有翻译器）
        if translater:
            print(f"Translating {paper_title}")
            retry_count, retry_seconds, NUM_RETRIES = 0, 60, 3
            while retry_count < NUM_RETRIES:
                try:
                    paper_abstract = translater.translate(paper_abstract)
                    break
                except Exception as e:
                    print(f"Error: {e}, retry after {retry_seconds}s.")
                    time.sleep(retry_seconds)
                    retry_count += 1
                    retry_seconds *= 2  # 指数退避
                finally:
                    if retry_count == NUM_RETRIES:
                        print(f"Translation failed after {NUM_RETRIES} attempts.")

        logging.info(f"Time = {update_time} title = {paper_title}")

        # 匹配代码链接（先标题、再论文 ID）
        repo_url = get_code_link(paper_title)
        if repo_url is None:
            repo_url = get_code_link(paper_id.split("v")[0])

        # 构造输出内容
        paper_abstract = paper_abstract.replace("\n", "")
        if repo_url:
            content[paper_id] = (
                "|**{}**|[{}]({})|**[link]({})**|{}|\n".format(
                    update_time, paper_title, paper_url, repo_url, paper_abstract
                )
            )
            content_to_web[paper_id] = (
                "- {}, Paper: [{}]({}), Code: **[{}]({})**, Abstract: {}\n".format(
                    update_time, paper_title, paper_url, repo_url, repo_url, paper_abstract
                )
            )
        else:
            content[paper_id] = (
                "|**{}**|[{}]({})|null|{}|\n".format(
                    update_time, paper_title, paper_url, paper_abstract
                )
            )
            content_to_web[paper_id] = (
                "- {}, Paper: [{}]({}), {}\n".format(
                    update_time, paper_title, paper_url, paper_abstract
                )
            )

        # 补充论文备注（若有）
        if result.comment:
            content_to_web[paper_id] += f", {result.comment}\n"

    return {topic: content}, {topic: content_to_web}


def update_paper_links(filename):
    """每周更新 JSON 文件中的论文代码链接"""

    def parse_arxiv_string(s):
        parts = s.split("|")
        date = parts[1].strip()
        title = parts[2].strip()
        paper_url = parts[3].strip()
        code = parts[4].strip()
        abstract = parts[5].strip()
        return date, title, paper_url, code, abstract

    with open(filename, "r") as f:
        json_data = json.loads(f.read() or "{}")

    for keywords, papers in json_data.items():
        logging.info(f"Updating links for: {keywords}")
        for paper_id, content in papers.items():
            date, title, paper_url, code, abstract = parse_arxiv_string(str(content))
            # 若代码链接为 null，尝试重新匹配
            if "|null|" in content:
                try:
                    repo_url = get_code_link(paper_id.split("v")[0])
                    if repo_url:
                        new_content = content.replace(
                            "|null|", f"|**[link]({repo_url})**|"
                        )
                        json_data[keywords][paper_id] = new_content
                        logging.info(f"Updated link for {paper_id}: {repo_url}")
                except Exception as e:
                    logging.error(f"Error updating {paper_id}: {e}")

    # 写回 JSON 文件
    with open(filename, "w") as f:
        json.dump(json_data, f, indent=2)


def update_json_file(filename, data_dict):
    """每日更新 JSON 文件内容"""
    with open(filename, "r") as f:
        json_data = json.loads(f.read() or "{}")

    # 合并新论文数据
    for data in data_dict:
        for topic, papers in data.items():
            if topic in json_data:
                json_data[topic].update(papers)
            else:
                json_data[topic] = papers

    with open(filename, "w") as f:
        json.dump(json_data, f, indent=2)


def json_to_md(
    filename,
    md_filename,
    task="",
    to_web=False,
    use_title=True,
    use_tc=True,
    use_b2t=True,
):
    """将 JSON 论文数据转为 Markdown"""

    def pretty_math(s: str) -> str:
        """美化 Markdown 中的数学公式"""
        match = re.search(r"\$.*\$", s)
        if not match:
            return s
        math_start, math_end = match.span()
        space_trail = " " if s[:math_start][-1] not in (" ", "*") else ""
        space_leading = " " if s[math_end:][0] not in (" ", "*") else ""
        return (
            s[:math_start]
            + f"{space_trail}${match.group()[1:-1].strip()}${space_leading}"
            + s[math_end:]
        )

    DateNow = datetime.date.today().strftime("%Y.%m.%d")

    # 读取 JSON 数据
    with open(filename, "r") as f:
        data = json.loads(f.read() or "{}")

    # 清空并重新写入 Markdown 文件
    with open(md_filename, "w+") as f:
        pass

    with open(md_filename, "a+") as f:
        # Web 页面头部（若需要）
        if use_title and to_web:
            f.write("---\nlayout: default\n---\n\n")

        # 日期标题
        f.write(f"## Updated on {DateNow}\n")
        f.write("> Usage instructions: [here](./docs/README.md#usage)\n\n")

        # 目录（若需要）
        if use_tc:
            f.write("<details>\n<summary>Table of Contents</summary>\n<ol>\n")
            for keyword in data:
                if data[keyword]:
                    kw = keyword.replace(" ", "-").lower()
                    f.write(f"  <li><a href=#{kw}>{keyword}</a></li>\n")
            f.write("</ol>\n</details>\n\n")

        # 按主题生成论文列表
        for keyword in data:
            papers = data[keyword]
            if not papers:
                continue
            f.write(f"## {keyword}\n\n")

            # Markdown 表格头
            if use_title:
                f.write(
                    "| Publish Date | Title | Code | Abstract |\n"
                    "|:---------|:-----------------------|:------|:-------------------------------------------------|\n"
                )

            # 按日期降序排列论文
            for _, content in sort_papers(papers).items():
                if content:
                    f.write(pretty_math(content))  # 美化数学公式

            f.write("\n")

            # 返回顶部链接（若需要）
            if use_b2t:
                top_anchor = f"#updated-on-{DateNow.replace('.', '')}"
                f.write(f'<p align=right>(<a href="{top_anchor}">back to top</a>)</p>\n\n')

    logging.info(f"{task} finished")


def demo(translater: Optional[Translater] = None, **config):
    """演示逻辑：获取论文、更新文件"""
    data_collector, data_collector_web = [], []
    keywords = config["kv"]
    max_results = config["max_results"]
    publish_readme = config["publish_readme"]
    publish_gitpage = config["publish_gitpage"]
    b_update = config["update_paper_links"]

    logging.info(f"Update Paper Link = {b_update}")
    if not b_update:
        logging.info("Getting daily papers...")
        for topic, query in keywords.items():
            logging.info(f"Topic: {topic}, Query: {query}")
            data, data_web = get_daily_papers(
                topic, query=query, max_results=max_results, translater=translater
            )
            data_collector.append(data)
            data_collector_web.append(data_web)
        logging.info("Daily papers fetched.")

    # 更新 README.md
    if publish_readme:
        json_file, md_file = config["json_readme_path"], config["md_readme_path"]
        if b_update:
            update_paper_links(json_file)
        else:
            update_json_file(json_file, data_collector)
        json_to_md(json_file, md_file, task="Update Readme")

    # 更新 GitPage 页面（docs/index.md）
    if publish_gitpage:
        json_file, md_file = config["json_gitpage_path"], config["md_gitpage_path"]
        if b_update:
            update_paper_links(json_file)
        else:
            update_json_file(json_file, data_collector)
        json_to_md(
            json_file,
            md_file,
            task="Update GitPage",
            to_web=True,
            use_tc=False,
            use_b2t=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="Configuration file path",
    )
    parser.add_argument(
        "--update_paper_links",
        action="store_true",
        help="Whether to update paper links",
    )
    parser.add_argument(
        "--google_api_key",
        type=str,
        default="",
        help="Google Gemini API Key",
    )
    args = parser.parse_args()

    # 加载配置并覆盖关键词（示例：直接写死多领域关键词）
    config = load_config(args.config_path)
    config["kv"] = {
        # 原有主题保留
        "多模态": (
            'abs:("Multi-modal Models" OR "Multimodal" OR "vision-language" '
            'OR "Vision Language Models" "Vision-and-Language Pre-training" '
            'OR "Multimodal Learning" OR "multimodal pretraining")'
            'AND abs:("model")'
        ),
        "生成模型": 'abs:("diffusion model" OR "text-to-video synthesis" OR "generative model")',
        "Transformer": (
            'abs:("self-attention" OR "cross-attention" OR "cross attention" '
            'OR "Sparse attention" OR "attention") AND abs:("transformer") '
        ),
        "大模型PEFT": (
            'abs:("PEFT" OR "parameter-efficient fine-tuning" '
            'OR "foundation model LoRA" OR "large language model adapter" OR "LLM adapter" '
            'OR "LoRA")'
        ),
        "大模型强化学习": (
            'abs:("reinforcement learning" OR "RLHF" '
            'OR "foundation model reinforcement learning from human feedback" '
            'OR "RLVR" OR "GRPO")'
            'AND abs:("model")'
        ),
        "大模型持续学习": (
            'abs:("Multimodal Large Language Models" OR "Large Language Models"'
            'OR "MLLM" OR "LLM" OR "VLM")'
            'AND abs:("continual learning" OR "continual pre-training")'
        )
    }
    config["update_paper_links"] = args.update_paper_links

    # 若提供 Google API Key，则初始化翻译器
    if args.google_api_key:
        translater = Translater(api_key=args.google_api_key)
        demo(translater, **config)
    else:
        demo(**config)
