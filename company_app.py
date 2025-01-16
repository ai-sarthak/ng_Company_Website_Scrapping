import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import os
import PyPDF2
import tempfile
import time
import scrapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime  # Ensure this line is present


# Function to read company data
def read_company_data(file_path):
    df = pd.read_csv(file_path)
    return df[['Company', 'Website', 'Person LinkedIn Url']].dropna()

# Function to scrape the website content with logging and retry mechanism
def scrape_website(url, max_retries=3):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    retries = 0
    start_time = time.time()
    log_entry = {
        'Website': url,
        'Company Name': '',
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
            response = requests.get(url, headers=headers, timeout=10, cookies=cookies_sent)
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
        except requests.RequestException as e:
            log_entry['Status'] = 'Error'
            log_entry['Description'] = str(e)

        retries += 1
        log_entry['Retries'] = retries
        time.sleep(2 ** retries)  # Exponential backoff

    return "", None, log_entry

# Function to scrape PDF content from website
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
                # Clean the PDF text: remove new lines and extra spaces
                pdf_text = ' '.join(pdf_text.split())  # This removes extra spaces and new lines
                pdf_texts.append(pdf_text)
                os.remove(temp_pdf_path)
            except requests.RequestException as e:
                print(f"Error downloading PDF {pdf_link}: {e}")
    return pdf_texts

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in range(len(reader.pages)):
                pdf_text += reader.pages[page].extract_text() or ''  # Ensure text is added even if None
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_path}: {e}")
    return pdf_text

# Function to generate a company profile
def generate_profile(company_name, url, linkedin_url, text, pdf_texts):
    profile = {
        "Company": company_name,
        "Website": url,
        "Person LinkedIn URL": linkedin_url,  # Changed to Personal LinkedIn URL
        "Website Text": text,
        "PDF Text": ' '.join(pdf_texts)  # Join PDF texts into a single string
    }
    return profile

# Function to process a single company
def process_company(row):
    url = row['Website']
    linkedin_url = row['Person LinkedIn Url']
    company_name = row['Company']  # Read company name from the input row

    print(f"Processing {company_name} at {url}...")
    text, soup, log_entry = scrape_website(url)
    pdf_texts = scrape_pdfs(soup, url)

    if text:
        profile = generate_profile(company_name, url, linkedin_url, text, pdf_texts)
        return profile, log_entry
    else:
        print(f"No content found for {url}.")
        return None, log_entry

# Function to analyze scraped data with LLM after scraping
def analyze_with_llm(profiles):
    for profile in profiles:
        print(f"Analyzing {profile['Company']} with LLM...")
        website_analysis = scrapping.Analyze_scrap(profile['Website Text'])
        pdf_analysis = scrapping.Analyze_scrap(profile['PDF Text'])
        profile['Signals'] = {
            "Website Analysis": website_analysis,
            "PDF Analysis": pdf_analysis
        }
    return profiles

# Main function to process companies with concurrency
def process_companies(file_path, output_file, log_file, num_threads=5):
    company_data = read_company_data(file_path)
    profiles = []
    logs = []  # List to store logs

    # Step 1: Scrape websites concurrently
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_company, row) for _, row in company_data.iterrows()]

        for future in as_completed(futures):
            profile, log_entry = future.result()
            if profile:
                profiles.append(profile)
            logs.append(log_entry)

    # Step 2: Analyze with LLM (sequentially after scraping to avoid rate limits)
    profiles = analyze_with_llm(profiles)

    # Step 3: Save profiles and logs to CSV
    if profiles:
        save_profiles_to_csv(output_file, profiles)
    if logs:
        save_logs_to_csv(log_file, logs)

# Function to save profiles to CSV
def save_profiles_to_csv(file_path, profiles):
    df = pd.DataFrame(profiles)
    df.to_csv(file_path, index=False, header=True)
    print(f"Saved {len(profiles)} profiles to {file_path}")

# Function to save logs to CSV
def save_logs_to_csv(file_path, logs):
    df = pd.DataFrame(logs)
    df.to_csv(file_path, index=False, header=True)
    print(f"Saved {len(logs)} logs to {file_path}")

# Example usage
input_file = 'testing - updated.csv'  # Replace with your actual file path
output_file = 'website_data.csv'  # Output file for profiles
log_file = 'Scraping_Logs.csv'  # Output file for logs
num_threads = 1  # Specify the number of concurrent threads here
process_companies(input_file, output_file, log_file, num_threads)
