import json
from web_search.webpage_utils import extract_text_from_urls, error_indicators
import re
import asyncio
from openai import OpenAI
import requests
from urllib.parse import urlencode

def deep_search(query, api_key="your_api_key", zone="your_zone_here"):
    encoded_query = urlencode({"q": query})

    url = f"https://www.bing.com/search?{encoded_query}&brd_json=1"

    api_url = "https://api.brightdata.com/request"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "zone": zone,
        "url": url,
        "format": "raw"
    }

    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    json_data = response.text
    data = json.loads(json_data)
    chunk_content_list = []
    for result in data['organic']:
        # title = result.get('title', '')
        snippet = result.get('description', '')
        chunk_content_list.append(snippet)
    data['chunk_content'] = chunk_content_list

    result = ''
    for item in data['chunk_content']:
        # item['snippet'] = formatted_documents
        result += item + '\n'
    return result.strip()

def deep_search_with_summarize(query, api_key="your_api_key", zone="your_zone_here"):
    encoded_query = urlencode({"q": query})

    url = f"https://www.bing.com/search?{encoded_query}&brd_json=1"

    api_url = "https://api.brightdata.com/request"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "zone": zone,
        "url": url,
        "format": "raw"
    }

    print("--------------------------------begin brightdata api search--------------------------------")

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(data))

        if response.status_code != 200:
            return "No relevant information found."
        
        search_results = json.loads(response.text)

        if isinstance(search_results, dict) and 'error' in search_results:
            return "No relevant information found."
    except Exception as e:
        print(f"Error during search query '{query}': {e}")
        return "No relevant information found."
    
    print("--------------------------------get bing search results--------------------------------")
    
    relevant_info = []
    if 'organic' in search_results:
        for result in search_results['organic']:
            doc_info = {
                'title': result.get('title', ''),
                'url': result.get('link', ''),
                'snippet': result.get('description', '')
            }
            relevant_info.append(doc_info)

    if len(relevant_info) == 0:
        return "No relevant information found."
    
    url_cache = {}
    urls_to_fetch = [doc_info['url'] for doc_info in relevant_info]
    snippets = [doc_info['snippet'] for doc_info in relevant_info]
    
    print("--------------------------------begin deep websearch--------------------------------")
    
    try:
        contents = extract_text_from_urls(urls_to_fetch, snippet=snippets)
        for url, content in zip(urls_to_fetch, contents):
            # Only cache content if it doesn't contain error indicators
            has_error = (any(indicator.lower() in content.lower() for indicator in error_indicators) and len(content.split()) < 64) or len(content) < 50 or len(content.split()) < 20
            if not has_error:
                url_cache[url] = content
    except Exception as e:
        print(f"Error fetching URLs: {e}")
    
    print("--------------------------------get websearch results--------------------------------")
    
    formatted_documents = ""
    not_nont_documents = []
    none_documents = []
    for i, doc_info in enumerate(relevant_info):
        url = doc_info['url']
        title = doc_info.get('title', '')
        snippet = doc_info.get('snippet', '')
        raw_context = url_cache.get(url, "")
        
        clean_title = re.sub('<[^<]+?>', '', title)
        clean_snippet = re.sub('<[^<]+?>', '', snippet)
        if raw_context:
            not_nont_documents.append(
                f"**Title:** {clean_title}\n**URL:** {url}\n**Snippet:** {clean_snippet}\n**Content:** {raw_context[:1500]}...\n\n" if len(raw_context) > 1500 else f"**Title:** {clean_title}\n**URL:** {url}\n**Snippet:** {clean_snippet}\n**Content:** {raw_context}\n\n"
            )
        else:
            none_documents.append(
                f"**Title:** {clean_title}\n**URL:** {url}\n**Snippet:** {clean_snippet}\n**Content:** No content available.\n\n"
            )
    if len(not_nont_documents) >= 5:
        for i in range(5):
            formatted_documents += f"**Document {i + 1}:**\n{not_nont_documents[i]}"
    else:
        for i in range(len(not_nont_documents)):
            formatted_documents += f"**Document {i + 1}:**\n{not_nont_documents[i]}"
        for i in range(5 - len(not_nont_documents)):
            formatted_documents += f"**Document {len(not_nont_documents) + i + 1}:**\n{none_documents[i]}"
    
    if formatted_documents == "":
        formatted_documents = "No relevant information found."
    else:
        formatted_documents = summarize_text(query, formatted_documents)
    
    return formatted_documents


def summarize_text(search_query, search_result):

    API_BASE_URL = "your_summary_api_base_url"  # replace with your actual API base URL
    MODEL_NAME = "Qwen2.5-7B-Instruct"
    client = OpenAI(
        api_key="empty",
        base_url=API_BASE_URL,
    )


    prompt = f"""You are a web explorer. Your task is to find relevant information for the search query.

    Based on the search results:
    - If the information answers the query, output it directly
    - If more information is needed:
    1. Search again: <|begin_search_query|>query<|end_search_query|>
    2. Access webpage content with information seeking intent using: <|begin_click_link|>URL<|end_click_link|> <|begin_click_intent|>intent<|end_click_intent|>
    - If you can't find any helpful information, output "No helpful information found."

    **Inputs:**

    **Search Query:**
    {search_query}

    **Search Results:**
    {search_result}

    Final Output Format:

    **Final Information:**

    [Factual information to the search query] or [No helpful information found.]

    Now you should analyze each web page and find helpful information for the search query "{search_query}"."""

    def sanitize_input(text):
        if isinstance(text, str):
            text = text.replace('\\', '\\\\')
            text = text.replace('{', '{{').replace('}', '}}')
        return text
    
    search_query = sanitize_input(search_query)
    search_result = sanitize_input(search_result)

    prompt = prompt.format(search_query=search_query, search_result=search_result)
    
    
    chat_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    return chat_response.choices[0].message.content






if __name__ == "__main__":

    extracted_info = ''
    import argparse
    parser = argparse.ArgumentParser(description="Deep Search")
    parser.add_argument('--key', type=str, default=None, help='The search query to use for deep search.')
    args = parser.parse_args()
    
    question = "三角形的勾股定理是什么"

    result = deep_search_with_summarize(
        question, api_key=args.key, zone="your_zone_here"  # replace with your actual zone
    )
    print('-------------------------------------')
    print(result)
    print('-------------------------------------')