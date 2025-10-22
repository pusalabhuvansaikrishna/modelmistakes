from bs4 import BeautifulSoup
import requests
import streamlit as st
import pandas as pd
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from nltk.metrics import edit_distance
from evaluation.normalize import Normalize
from datetime import datetime
from langdetect import detect
from language_dict import language_map
import redis
import json

r=redis.Redis(host="localhost",port=6379,decode_responses=True)

def detect_language(text):
    code=detect(text)
    return language_map.get(code.lower())

def get_data(url:str):
    headings=[]
    paragraphs=[]
    if ("https" or "http") in url:
        response=requests.get(url)
        if "200" in str(response.status_code):
            soup=BeautifulSoup(response.text, "html.parser")
            page_layout=soup.find("div",class_="col")
            page_no=(page_layout.find("a").text).lstrip("#")
            gt=(((soup.find("p")).text))
            parent=soup.find("div", class_="parent")
            children=parent.find_all("div",class_="child")
            for child in children:
                heading=(child.find("div", class_="heading").text).strip()
                headings.append(heading)
                try:
                    paragraph_container=child.find("div", style="white-space: pre-wrap; font-size: smaller;")
                    paragraph=paragraph_container.find("p").text
                    paragraphs.append(paragraph)
                except:
                    continue
            headings.remove("Original Image")
            return page_no,gt,headings, paragraphs
        else:
            return ((str(response.status_code)).lstrip("<")).rstrip(">")
    else:
        return "Url is not in proper format, Make sure it is in http(s)://example.com/..."

def process_url(url):
    page_no, gt, headings, paragraphs=get_data(url)
    return page_no,{"ground_truth":gt,**dict(zip(headings, paragraphs))}

def process_file_parallel(df,max_workers=4):
    urls=df['URL'].unique()
    processed_data={}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures={}
        for url in urls:
            if r.exists(url):
                stored_data=r.get(url)
                stored_dict=json.loads(stored_data)
                page_no=stored_dict['page_no']
                page_dict=stored_dict['page_dict']
                processed_data[page_no]=page_dict
            else:
                futures[executor.submit(process_url,url)]=url

        for future in as_completed(futures):
            url=futures[future]
            page_no,page_dict=future.result()
            processed_data[page_no]=page_dict
            to_store={"page_no":page_no,"page_dict":page_dict}
            r.set(url, json.dumps(to_store))
    return processed_data

def clean(inp: str) -> str:
    """Clean text: reduce whitespaces, normalize line breaks."""
    while '\n\n' in inp:
        inp = inp.replace('\n\n', '\n')
    while '\r\n' in inp:
        inp = inp.replace('\r\n', '\n')
    while '\r' in inp:
        inp = inp.replace('\r', '')
    while '\n' in inp:
        inp = inp.replace('\n', ' ')
    while '  ' in inp:
        inp = inp.replace('  ', ' ')
    return inp.strip()

def edit_distance(s1: str, s2: str) -> int:
    """Compute character-level Levenshtein edit distance."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def remove_stray_tokens(pred_text: str, stray_tokens=None) -> str:
    """Remove unwanted tokens like 'del' from predicted text."""
    if stray_tokens is None:
        stray_tokens = ["del"]
    for token in stray_tokens:
        pred_text = pred_text.replace(token, "")
    return pred_text

def merge_splits(gt_words, pred_words):
    """Merge consecutive predicted words if they match any GT word ignoring spaces."""
    merged_pred = []
    i = 0
    while i < len(pred_words):
        current = pred_words[i]
        j = i + 1
        # Try merging next word(s)
        while j <= len(pred_words):
            candidate = "".join(pred_words[i:j])
            if candidate in gt_words:
                current = candidate
                i = j - 1
                break
            j += 1
        merged_pred.append(current)
        i += 1
    return merged_pred

def find_word_differences(gt_text: str, pred_text: str, language: str):
    """Compare GT and predicted text word by word and return mismatches with edit distance."""
    # Clean and normalize
    gt_text = clean(gt_text)
    pred_text = clean(pred_text)

    gt_text = Normalize(language).run(gt_text)
    pred_text = Normalize(language).run(pred_text)

    # Remove stray tokens
    pred_text = remove_stray_tokens(pred_text)

    # Split into words
    gt_words = gt_text.split()
    pred_words = pred_text.split()

    # Merge split words in prediction
    pred_words = merge_splits(gt_words, pred_words)

    # Compare words and compute edit distance
    results = []
    max_len = max(len(gt_words), len(pred_words))
    for i in range(max_len):
        gt_word = gt_words[i] if i < len(gt_words) else ""
        pred_word = pred_words[i] if i < len(pred_words) else ""
        if gt_word != pred_word:
            distance = edit_distance(gt_word, pred_word)
            results.append({
                "GroundTruth": gt_word,
                "Predicted": pred_word,
                "EditDistance": distance
            })

    return results

def process_heading(a:str):
    req=a.split("Text Output ")[1]
    return req.lstrip("(")[:-1]
def process_erors(errors):
    ground_truth=[]
    predicted=[]
    edit_dist=[]
    for error in iter(errors):
        ground_truth.append(error["GroundTruth"])
        predicted.append(error['Predicted'])
        edit_dist.append(error['EditDistance'])
    return ground_truth, predicted, edit_dist

def identify_langage(text):
    prediction=model.predict(text)
    return prediction.language
def find_errors(data):
    error_data={"Page ID":[], "Model Name":[], "Ground Truth":[], "OCR Text":[], "Edit Distance":[]}
    pages=data.keys()
    for page in pages:
        groundtruth=data[page]['ground_truth']
        for model,text in data[page].items():
            if model=="ground_truth":
                continue
            else:
                lang=detect_language(groundtruth)
                for i in find_word_differences(groundtruth,data[page][model],lang) or []:
                    error_data['Page ID'].append(page)
                    error_data['Model Name'].append(process_heading(model))
                    error_data['Ground Truth'].append(i['GroundTruth'])
                    error_data['OCR Text'].append(i['Predicted'])
                    error_data['Edit Distance'].append(i['EditDistance'])

    return error_data


def excel_new(data):
    timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name=f'Result_{timestamp}'
    reports_folder=os.path.join(os.getcwd(),"Reports")
    if not os.path.exists(reports_folder):
        os.makedirs(reports_folder)
    df=pd.DataFrame(data)
    file_path = os.path.join(os.getcwd(), "Reports", f"{file_name}.xlsx")
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df.to_excel(writer,sheet_name="Error Data", index=True)

    return file_name

st.set_page_config(page_title="Find What Went Wrong by Models",page_icon='🤔',layout="wide",initial_sidebar_state=None,menu_items=None)
st.title("🤔 Find What Went Wrong by the Models!")
st.info("""This tool automates OCR validation by cross-checking OCR model outputs with ground-truth text collected from the Testing Portal.""", icon="ℹ️",width="stretch")
uploaded_file=st.file_uploader("Upload the Test Report CSV",type="csv",accept_multiple_files=False,label_visibility="visible",help="Currently Accepting Only CSV Files")
if uploaded_file:
    original_df=pd.read_csv(uploaded_file)
    with st.status("Processing the report...",expanded=True,state='running') as status:
        #Extract the Information from webpages to dict.
        st.write("Extracting the information...")
        start=time.time()
        dict_data=process_file_parallel(original_df)
        st.badge(str(time.time()-start),icon=":material/timer:", color="blue")

        #Finding the error in the errors in the text
        st.write("Finding the errors...")
        start=time.time()
        error_data=find_errors(dict_data)
        st.badge(str(time.time()-start), icon=":material/timer:", color="blue")

        st.write("Preparing Excel to Download!")
        start=time.time()
        name=excel_new(error_data)
        st.badge(str(time.time()-start),icon=":material/timer:", color="blue")
        status.update(label="Ready to Download the Report!",state="complete",expanded=True)

        col1,col2,col3=st.columns(3)
        with col2:
            file_path = os.path.join(os.getcwd(), "Reports", f"{name}.xlsx")
            with open(file_path,"rb") as f:
                file_bytes=f.read()
            st.download_button("Download",file_bytes,file_name=f'{name}.xlsx',on_click="ignore",type="primary", icon=':material/download:',width='content')

