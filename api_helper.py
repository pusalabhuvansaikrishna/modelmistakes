import re
import requests
def process_data(marked_data):
    marked_data = re.sub(r'<mark[^>]*>del</mark>', '', marked_data)
    cleaned = re.sub(r'</?mark[^>]*>', '', marked_data)
    paragraph = ' '.join(cleaned.split())
    return paragraph



def get_data_api(url):
    headings=[]
    paragraphs=[]
    response=requests.get(url)
    if response.status_code != 200:
        print("Failed to fetch Data:", url)
        return None, None, [], []

    data = response.json()

    page_id = data['id']
    gt = data['gt']

    for item in data['ocr_list']:
        if item['layout_model'] == "GT":
            model_name = item['layout_model'] + "/" + item['ocr_model']
            paragraph = process_data(item['text'])

            headings.append(model_name)
            paragraphs.append(paragraph)

    return page_id, gt, headings, paragraphs