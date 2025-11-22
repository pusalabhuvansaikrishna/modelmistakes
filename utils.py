import main
from api_helper import get_data_api
def process_url(url):
    page_no, gt, headings, paragraphs=get_data_api(url)
    page_dict={
        "ground_truth":gt,
        **dict(zip(headings,paragraphs))
    }
    return page_no,page_dict