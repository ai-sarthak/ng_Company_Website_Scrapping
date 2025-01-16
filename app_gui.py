import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import PyPDF2
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
import shutil
from urllib.parse import urlparse
from datetime import datetime
import scrapping  # Import your scrapping module to use the Analyze_scrap function

# Function to read company data
def read_company_data(file_path):
    df = pd.read_csv(file_path)
    return df[['Company', 'Website', 'Person LinkedIn Url']].dropna()

# Function to scrape the website content with timeout handling and detailed logs
def scrape_website(url, company_name, max_retries=3, timeout=20):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    retries = 0
    start_time = time.time()
    
    # Initial log entry setup
    log_entry = {
        'Website': url,
        'Company Name': company_name,
        'Domain': urlparse(url).netloc,
        'Status': None,
        'Description': '',
        'Retries': 0,
        'Time of Attempt': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Response Code': None,
        'Response Time': None,
        'Headers Sent': str(headers),
        'Headers Received': '',
        'Page Title': '',
        'Content Length': 0,
        'Number of Links': 0,
        'Number of PDFs': 0,
        'First PDF URL': '',
        'Redirected URL': '',
        'User-Agent': headers['User-Agent'],
        'Cookies Sent': '',
        'Cookies Received': '',
    }

    cookies_sent = {}
    
    while retries < max_retries:
        try:
            response = requests.get(url, headers=headers, timeout=timeout, cookies=cookies_sent)
            log_entry['Response Code'] = response.status_code
            log_entry['Response Time'] = round(time.time() - start_time, 2)  # in seconds
            log_entry['Headers Received'] = str(response.headers)
            log_entry['Cookies Received'] = str(response.cookies)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                text = soup.get_text(separator=' ')
                
                # Clean the website text: remove new lines and extra spaces
                text = ' '.join(text.split())  # This removes extra spaces and new lines

                log_entry['Page Title'] = soup.title.string if soup.title else 'N/A'
                log_entry['Content Length'] = len(text)
                log_entry['Number of Links'] = len(soup.find_all('a'))
                pdf_links = [link.get('href') for link in soup.find_all('a', href=True) if link.get('href').endswith('.pdf')]
                log_entry['Number of PDFs'] = len(pdf_links)
                log_entry['First PDF URL'] = pdf_links[0] if pdf_links else 'N/A'
                log_entry['Redirected URL'] = response.url if response.history else 'N/A'

                log_entry['Status'] = 'Success'
                log_entry['Description'] = 'Website scraped successfully'
                return text, soup, log_entry
            else:
                log_entry['Status'] = 'Failed'
                log_entry['Description'] = f"Received {response.status_code} response"
        except requests.exceptions.Timeout:
            log_entry['Status'] = 'Error'
            log_entry['Description'] = 'Request timed out'
            log_entry['Retries'] = retries
        except requests.RequestException as e:
            log_entry['Status'] = 'Error'
            log_entry['Description'] = str(e)

        retries += 1
        log_entry['Retries'] = retries
        time.sleep(2 ** retries)  # Exponential backoff

    return "", None, log_entry

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                pdf_text += page.extract_text() or ''
    except Exception:
        pass
    return pdf_text

# Function to scrape PDFs
def scrape_pdfs(soup, base_url):
    pdf_texts = []
    if soup:
        pdf_links = [link.get('href') for link in soup.find_all('a', href=True) if link.get('href').endswith('.pdf')]
        for pdf_link in pdf_links:
            if not pdf_link.startswith('http'):
                pdf_link = base_url + pdf_link
            try:
                pdf_response = requests.get(pdf_link)
                pdf_response.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
                    temp_pdf.write(pdf_response.content)
                    temp_pdf_path = temp_pdf.name
                pdf_text = extract_text_from_pdf(temp_pdf_path)
                pdf_texts.append(pdf_text)
                os.remove(temp_pdf_path)
            except requests.RequestException:
                pass
    return pdf_texts

# Function to generate profile
def generate_profile(company_name, url, linkedin_url, text, pdf_texts):
    profile = {
        "Company": company_name,
        "Website": url,
        "Person LinkedIn URL": linkedin_url,  # Changed to Personal LinkedIn URL
        "Website Text": text,
        "PDF Text": ' '.join(pdf_texts)  # Join PDF texts into a single string
    }
    return profile

# Function to analyze scraped data with LLM
def analyze_with_llm(profiles):
    for profile in profiles:
        st.write(f"Analyzing {profile['Company']} with LLM...")
        website_analysis = scrapping.Analyze_scrap(profile['Website Text'])
        pdf_analysis = scrapping.Analyze_scrap(profile['PDF Text'])
        profile['Signals'] = {
            "Website Analysis": website_analysis,
            "PDF Analysis": pdf_analysis
        }
    return profiles

# Main function to process companies with concurrency
def process_companies(file_path, num_threads):
    company_data = read_company_data(file_path)
    profiles = []
    logs = []  # Initialize the logs list here
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_company, row, logs): row for _, row in company_data.iterrows()}
        for i, future in enumerate(as_completed(futures)):
            row = futures[future]
            company_name = row['Company']
            try:
                profile = future.result()
                if profile:
                    profiles.append(profile)
            except Exception as e:
                st.write(f"Error processing {company_name}: {e}")
            finally:
                st.session_state.progress_bar.progress((i + 1) / len(company_data))
                st.write(f"{len(company_data) - (i + 1)} companies remaining")
    
    # After scraping, analyze the profiles with LLM
    profiles = analyze_with_llm(profiles)

    return profiles, logs

# Function to process a single company
def process_company(row, logs):  # Add logs as argument
    url = row['Website']
    linkedin_url = row['Person LinkedIn Url']
    company_name = row['Company']

    st.write(f"Processing {company_name} at {url}...")
    text, soup, log_entry = scrape_website(url, company_name)  # Scraping with timeout handling and logging
    pdf_texts = scrape_pdfs(soup, url)

    # Log the entry into the logs list
    logs.append(log_entry)  # Append the log entry to logs
    
    if text:
        profile = generate_profile(company_name, url, linkedin_url, text, pdf_texts)
        return profile
    else:
        st.write(f"No content found for {url}.")
        return None


# Function to zip the profiles and logs together
def create_zip_file(profiles, logs):
    # Save output files
    output_file = "output_profiles.csv"
    log_file = "scraping_logs.csv"
    pd.DataFrame(profiles).to_csv(output_file, index=False)
    pd.DataFrame(logs).to_csv(log_file, index=False)

    # Create a zip file
    zip_file = "scraping_output.zip"
    with zipfile.ZipFile(zip_file, 'w') as zipf:
        zipf.write(output_file)
        zipf.write(log_file)

    # Clean up the CSV files after zipping
    os.remove(output_file)
    os.remove(log_file)

    return zip_file

# Streamlit Interface
st.set_page_config(page_title="Company Website Scraper and Analyzer", layout="wide")
st.title("Company Website Analyzer Tool")

# Sidebar: API Key Inputs
st.sidebar.header("Please Add your gemini-1.5-flash LLM API Keys")
num_api_keys = st.sidebar.number_input("How many API keys?", min_value=1, max_value=5, step=1, value=1)
api_keys = [st.sidebar.text_input(f"API Key {i+1}", type="password") for i in range(num_api_keys)]
st.sidebar.write("Stored API keys:", api_keys)

# Sidebar: Number of Threads
num_threads = st.sidebar.number_input("Number of Threads ", min_value=1, max_value=20, step=1, value=5)

# Center: File Upload and Progress Bar
uploaded_file = st.file_uploader("Upload a CSV file with company data", type="csv")
if uploaded_file:
    if "progress_bar" not in st.session_state:
        st.session_state.progress_bar = None

    if st.button("Start Processing"):
        st.session_state.progress_bar = st.progress(0)
        profiles, logs = process_companies(uploaded_file, num_threads)
        st.session_state.progress_bar.progress(1.0)

        # Create a zip file containing both the profiles and logs
        zip_file = create_zip_file(profiles, logs)

        # Store the zip file path in session state
        st.session_state.zip_file = zip_file

        # Right Section: Show Output Files
        with st.sidebar:
            st.header("Output Files")
            if "zip_file" in st.session_state:
                with open(st.session_state.zip_file, "rb") as file:
                    st.download_button("Download All Files", data=file.read(), file_name=st.session_state.zip_file)
            st.write("Files saved to:", zip_file)
