import streamlit as st
import pandas as pd
import re
from io import BytesIO
import os

# --- AI INTEGRATION SETUP ---
# Ensure you have installed google-genai: pip install google-genai
from google import genai
from google.genai import types 
from google.genai.errors import APIError

@st.cache_resource(show_spinner=False)
def get_gemini_client():
    """Initializes the Gemini client using the key from Streamlit Secrets."""
    # NOTE: This assumes GEMINI_API_KEY is configured in Streamlit Secrets
    if "GEMINI_API_KEY" not in st.secrets:
        # st.error("GEMINI_API_KEY not found in Streamlit secrets.")
        return None
    
    api_key = st.secrets["GEMINI_API_KEY"]
    os.environ["GEMINI_API_KEY"] = api_key
    
    try:
        # Check if the environment variable is set before creating the client
        if os.getenv("GEMINI_API_KEY"):
            return genai.Client()
        else:
            return None
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return None

client = get_gemini_client()
# --- END AI INTEGRATION SETUP ---


# --- 2. HELPER FUNCTIONS ---

def clean_value(value):
    """
    Cleans numeric values by handling SA (comma for decimal, space/dot for thousands) format.
    Kept as a safety net for the AI's JSON output.
    """
    if not isinstance(value, str): 
        # Handle direct float/int from AI's JSON output
        if isinstance(value, (int, float)):
            return float(value)
        return None
        
    value = str(value).strip().replace('\n', '').replace('\r', '') 
    
    # 1. Remove currency symbols and merge spaces between digits
    value = re.sub(r'[R$]', '', value, flags=re.IGNORECASE)
    value = re.sub(r'(\d)\s+(\d)', r'\1\2', value) 

    # 2. Handle South African formatting (1 000,00 or 1.000,00)
    if ',' in value:
        value = value.replace('.', '').replace(' ', '')
    else: # Handle standard dot-as-decimal format if no comma is present
        value = value.replace(' ', '')

    # 3. Replace the South African decimal comma with a standard dot
    value = value.replace(',', '.')
    
    # 4. Clean up formatting indicators (Dr/Cr)
    value = value.replace('Cr', '').replace('Dr', '-').strip()
    
    # 5. Final aggressive cleanup to remove non-numeric/non-dot/non-sign characters
    value = re.sub(r'[^\d\.\-]+', '', value) 
        
    try:
        if re.match(r'^-?\d*\.?\d+$', value):
            return float(value)
        return None
    except:
        return None

def clean_description_for_xero(description):
    """Cleans up transaction descriptions for easy Xero reconciliation."""
    if not isinstance(description, str): return ""
        
    description = description.strip()
    
    # Remove common reference/date patterns left over by extraction
    description = re.sub(r'\s*\d{6}\s+\d{4}\s+\d{2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', '', description, flags=re.IGNORECASE)
    description = re.sub(r'(?:Ref\s*|Reference\s*|No\s*|Nr\s*|ID\s*):\s*[\w\d\-]+', '', description, flags=re.IGNORECASE)
    description = re.sub(r'Serial:\d+/\d+', '', description)

    # Remove common transaction type prefixes
    description = re.sub(r'(?:POS Purchase|ATM Withdrawal|Immediate Payment|Internet Pmt To|Teller Transfer Debit|Direct Credit|EFT|IB Payment)\s*', '', description, flags=re.IGNORECASE)
    
    description = re.sub(r'\s{2,}', ' ', description).strip(' -').strip()
    
    return description

# --- 4. CORE EXTRACTION LOGIC (GEMINI ONLY - WITH PRECISION COLUMN EXCLUSION) ---

def gemini_extract_from_pdf(pdf_file_path: BytesIO, file_name: str) -> tuple[pd.DataFrame, str | None]:
    """
    PRIMARY METHOD: Uses Gemini's vision capability for extraction based on strict JSON rules, 
    focusing on excluding the dedicated Fees (R) column and extracting the StatementYear.
    Returns a DataFrame and the extracted year (as a string, or None on failure).
    """
    st.info("🔄 **Initiating Gemini AI Extraction...** (Extracting Year and Transactions)")
    if not client:
        return pd.DataFrame(), None 

    try:
        pdf_part = types.Part.from_bytes(
            data=pdf_file_path.getvalue(),
            mime_type='application/pdf'
        )
        
        # --- THE CRITICAL, REFINED PROMPT (UPDATED FOR DYNAMIC YEAR EXTRACTION) ---
        prompt = f"""
            You are a South African accounting assistant. Your task is to extract the **statement year** and all **transaction lines** from the provided bank statement PDF ({file_name}). 
            The output **must be a single JSON object**.

            **Required Output Structure (Mandatory):**
            1.  'StatementYear': The four-digit year (e.g., "2025") found in the 'Statement Period' or 'Statement Date' section at the top of the statement.
            2.  'Transactions': A list of all transaction lines, where each item is a transaction object.

            **ABSOLUTE CRITICAL INSTRUCTION (MUST FOLLOW FOR TRANSACTIONS):**
            1.  **Fee Column Exclusion (The ONLY Exclusion):** If the statement layout shows separate columns for 'Fees (R)', 'Debits (R)', and 'Credits (R)', you MUST:
                * **IGNORE** the value in the dedicated **'Fees (R)' column** completely.
                * Extract the transaction 'Amount' **ONLY** from the 'Debits (R)' or 'Credits (R)' column.
            2.  **Inclusion of ALL Other Lines:** You MUST **INCLUDE** every transaction line item that has an amount in the 'Debits (R)' or 'Credits (R)' column.

            **Transaction Required Columns:**
            1.  'Date': The transaction date (in Day Month format, e.g., '01 Sep', '16 Mar').
            2.  'Description': A clean, single-line description of the transaction.
            3.  'Amount': The Rand amount from the Debit/Credit column (Debits must be negative, Credits positive).

            **Example JSON Output (Mandatory format):**
            {{
                "StatementYear": "2025",
                "Transactions": [
                    {{
                        "Date": "02/06",
                        "Description": "ATM/SSD WITHDRAWAL FEE",
                        "Amount": -100.00
                    }},
                    {{
                        "Date": "16 Mar",
                        "Description": "IMMEDIATE PAYMENT FAWZIA BAYAT",
                        "Amount": -2500.00
                    }}
                ]
            }}
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, pdf_part],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )

        import json
        
        json_text = response.text.strip().strip('```json').strip('```')
        data = json.loads(json_text)
        
        if isinstance(data, dict) and 'StatementYear' in data and 'Transactions' in data and isinstance(data['Transactions'], list):
            statement_year = str(data['StatementYear']).strip()
            
            st.success(f"Gemini AI Extraction successful! Year **{statement_year}** extracted with {len(data['Transactions'])} transactions.")
            return pd.DataFrame(data['Transactions']), statement_year
        else:
            st.error("Gemini extracted a result, but it was not the expected JSON structure (missing StatementYear or Transactions list).")
            return pd.DataFrame(), None
            
    except APIError as e:
        st.error(f"Gemini API Error: {e}. Check your API key or contact support.")
        return pd.DataFrame(), None
    except Exception as e:
        st.error(f"Gemini AI Extraction failed due to an unexpected error. Error: {e}")
        return pd.DataFrame(), None


def parse_pdf_data(pdf_file_path, file_name):
    """Core function: Uses Gemini AI for dynamic extraction, returning DataFrame and Year."""
    
    pdf_file_path.seek(0)
    
    # Updated to capture both the DataFrame and the extracted year
    df_gemini, statement_year = gemini_extract_from_pdf(pdf_file_path, file_name)
    
    if not df_gemini.empty and statement_year:
        required_cols = ['Date', 'Description', 'Amount']
        if not all(col in df_gemini.columns for col in required_cols):
             st.error("AI output is missing required columns (Date, Description, Amount) in the transaction list.")
             return pd.DataFrame(), None

        df_gemini['Date'] = df_gemini['Date'].astype(str)
        df_gemini['Description'] = df_gemini['Description'].astype(str)
        
        df_gemini['Amount'] = df_gemini['Amount'].apply(lambda x: clean_value(x)) 
        df_gemini.dropna(subset=['Amount'], inplace=True)
        
        if not df_gemini.empty:
            # Return the processed DataFrame and the extracted year
            return df_gemini[['Date', 'Description', 'Amount']], statement_year

    st.error(f"Gemini AI extraction failed for {file_name}. No data or year extracted.")
    return pd.DataFrame(), None


# --- 5. STREAMLIT APP LOGIC ---

if 'uploaded_files' not in st.session_state:
    st.session_state['uploaded_files'] = []

st.set_page_config(page_title="🇿🇦 Free SA Bank Statement to CSV Converter (AI Column-Only Filter)", layout="wide")

st.title("🇿🇦 SA Bank Statement PDF to CSV Converter (AI-Only - Dynamic Year)")
st.markdown("""
    ### Now using **Gemini AI exclusively** to **dynamically extract the statement year** and transactions, filtering only the dedicated Fees (R) column.
    ---
""")

if client:
    st.sidebar.success("Gemini AI Engine: **Active** ✅ (Exclusive Use - **Dynamic Year Extraction**)")
else:
    st.sidebar.warning("Gemini AI Engine: **Inactive** 🛑. Please set **GEMINI_API_KEY** in Streamlit Secrets.")


uploaded_files = st.file_uploader(
    "Upload your bank statement PDF files (Multiple files supported)",
    type=["pdf"],
    accept_multiple_files=True,
    key="unique_pdf_uploader_fixed" 
)

# --- PROCESSING STARTS HERE ---
if uploaded_files:
    st.subheader("Processing Files...")
    
    all_df = []
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        st.markdown(f"**Processing:** `{file_name}`")
        
        pdf_data = BytesIO(uploaded_file.read())
        
        # Capture both the DataFrame and the dynamically extracted year
        df_transactions, statement_year = parse_pdf_data(pdf_data, file_name)

        if not df_transactions.empty and 'Amount' in df_transactions.columns and statement_year:
            
            # The dynamically extracted year is now used for standardization
            current_year = statement_year 
            
            # Apply final cleaning and formatting
            df_transactions['Description'] = df_transactions['Description'].apply(clean_description_for_xero)
            
            df_final = df_transactions.rename(columns={
                'Date': 'Date',
                'Description': 'Description',
                'Amount': 'Amount'
            })
            
            # --- START: DATE FIX IMPLEMENTATION (Using dynamic year) ---
            try:
                # 1. Clean the date string 
                df_final['Date_Raw'] = df_final['Date'].astype(str).str.strip()

                # 2. Append the correct year to the extracted date (e.g., '01 Sep' -> '01 Sep 2025')
                df_final['Date_With_Year'] = df_final['Date_Raw'] + ' ' + current_year

                # 3. Attempt to parse the date using the explicit 'Day AbbreviatedMonth Year' format, which is common.
                df_final['Date_Parsed'] = pd.to_datetime(
                    df_final['Date_With_Year'], 
                    format='%d %b %Y', 
                    errors='coerce'
                )

                # 4. Handle cases where the AI may have output the date in a standard format or failed step 3
                failed_parsing = df_final['Date_Parsed'].isna()
                if failed_parsing.any():
                    # Fallback to general dayfirst parsing on the original raw date
                    df_final.loc[failed_parsing, 'Date_Parsed'] = pd.to_datetime(
                        df_final.loc[failed_parsing, 'Date_Raw'], 
                        errors='coerce', 
                        dayfirst=True 
                    )
                
                # 5. Format and update the final 'Date' column
                df_final['Date'] = df_final['Date_Parsed'].dt.strftime('%d/%m/%Y')
                
                # Drop rows where date parsing still failed
                df_final.dropna(subset=['Date'], inplace=True)
                
            except Exception as e:
                st.warning(f"Could not standardize dates for {file_name}. Dates remain in raw format. Error: {e}")
            # --- END: DATE FIX IMPLEMENTATION ---
            
            # Final structure: Date, Description, Amount
            df_xero = pd.DataFrame({
                'Date': df_final['Date'].fillna(''),
                'Description': df_final['Description'].astype(str),
                'Amount': df_final['Amount'].round(2), 
            })
            
            # Ensure the order is exactly Date, Description, Amount
            df_xero = df_xero[['Date', 'Description', 'Amount']]
            
            df_xero.dropna(subset=['Date', 'Amount'], inplace=True)
            
            all_df.append(df_xero)
            
            st.success(f"Successfully extracted {len(df_xero)} transactions from {file_name} (Year: {statement_year})")

    
    # --- 6. COMBINE AND DOWNLOAD ---

    if all_df:
        final_combined_df = pd.concat(all_df, ignore_index=True)
        
        st.markdown("---")
        st.subheader("✅ All Transactions Combined and Ready for Download (Fees Column Excluded, Year Dynamic)")
        
        st.dataframe(final_combined_df)
        
        # Convert DataFrame to CSV for download
        csv_output = final_combined_df.to_csv(index=False, sep=',', encoding='utf-8')
        st.download_button(
            label="⬇️ Download Column-Filtered CSV File",
            data=csv_output,
            file_name="SA_Bank_Statements_Dynamic_Year_Export.csv",
            mime="text/csv"
        )
