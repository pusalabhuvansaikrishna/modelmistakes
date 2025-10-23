def process_url(url):
    page_no, gt, headings, paragraphs=get_data(url)
    return page_no,{"ground_truth":gt,**dict(zip(headings, paragraphs))}