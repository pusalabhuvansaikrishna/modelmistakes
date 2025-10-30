import main
def process_url(url):
    page_no, gt, headings, paragraphs=main.get_data_api(url)
    return page_no,{"ground_truth":gt,**dict(zip(headings, paragraphs))}