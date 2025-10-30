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
    if response.status_code==200:
        data=response.json()
        page_id=data['id']
        gt=data['gt']
        for i in data['ocr_list']:
            model_name=i['layout_model']+"/"+i['ocr_model']
            paragraph=process_data(i['text'])
            headings.append(model_name)
            paragraphs.append(paragraph)
        return page_id, gt, headings, paragraphs
    else:
        print("Failed to fetch Data")

