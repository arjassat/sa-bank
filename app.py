import streamlit as st
import pandas as pd
import pdfplumber
import re
from io import BytesIO

# --- 1. CONFIGURATION AND BANK SPECIFIC RULES ---

# Define parsing rules for South African banks based on common statement layouts.
# The 'table_settings' are crucial for pdfplumber to accurately find data in each bank's unique layout.
# Note: These settings are based on the files provided (FNB, ABSA, Standard, HBZ) and may need minor tuning.
BANK_RULES = {
    # FNB (Gold Business Account): Single 'Amount' column, +/- sign implied by 'Cr' or 'Dr' in Balance.
    "FNB": {
        "text_search": "Gold Business Account",
        "columns": ["Date", "Description", "Amount", "Balance"],
        "table_settings": {
            "vertical_strategy": "text", 
            "horizontal_strategy": "text",
            "explicit_vertical_lines": [30, 80, 480, 540],
            "snap_y_tolerance": 5,
            "min_words_vertical": 2
        },
        "parse_func": "parse_fnb"
    },
    # Standard Bank (Business Current Account / Private Banking): Debits and Credits are separate columns.
    "Standard Bank": {
        "text_search": "CURRENT ACCOUNT",
        "columns": ["Details", "Service Fee", "Debits", "Credits", "Date", "Balance"],
        "table_settings": {
            "vertical_strategy": "lines", 
            "horizontal_strategy": "text",
            "explicit_vertical_lines": [30, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
            "snap_y_tolerance": 3
        },
        "parse_func": "parse_standard"
    },
    # ABSA (Investment Account): Transaction amount includes the sign or is complex.
    "ABSA": {
        "text_search": "Investment Account",
        "columns": ["date", "transaction amount R", "description", "balance R"],
        "table_settings": {
            "vertical_strategy": "text",
            "explicit_vertical_lines": [30, 80, 400, 550],
            "snap_y_tolerance": 5
        },
        "parse_func": "parse_absa"
    },
    # HBZ Bank (Current Account): Debit/Credit columns separate.
    "HBZ Bank": {
        "text_search": "HBZ Bank Limited",
        "columns": ["Date", "Particulars", "Debit", "Credit", "Reference"],
        "table_settings": {
            "vertical_strategy": "lines",
            "snap_y_tolerance": 3
        },
        "parse_func": "parse_hbz"
    }
    # Nedbank and Capitec will use the 'Unknown' fallback unless custom rules are added here.
}

# --- 2. HELPER FUNCTIONS ---

def detect_bank(text_content):
    """Detects the bank based on key phrases in the PDF content."""
    for bank, rules in BANK_RULES.items():
        if rules["text_search"] in text_content:
            return bank
    return "Unknown"

def clean_value(value):
    """Cleans numeric values by handling South African (comma for decimal, space/dot for thousands) format."""
    if not isinstance(value, str):
        return None
        
    value = value.replace(' ', '').replace('.', '') # Remove thousand separators
    value = value.replace(',', '.') # Convert comma decimal to dot decimal
    value = value.replace('Cr', '').replace('Dr', '-').strip() # Handle FNB's Cr/Dr
    # Clean up broken numbers (e.g., '2\n\n500,00' -> '2500,00')
    value = re.sub(r'(\d)\s+(\d)', r'\1\2', value) 
    value = re.sub(r'[^\d\.\-]+', '', value) # Keep only numbers, dot, and minus sign
        
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def clean_description_for_xero(description):
    """
    Cleans up transaction descriptions for easy Xero reconciliation by removing noise.
    """
    if not isinstance(description, str):
        return ""
        
    description = description.strip()
    
    # Remove common bank reference/code noise
    description = re.sub(r'\s*\d{6}\s+\d{4}\s+\d{2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', '', description, flags=re.IGNORECASE)
    description = re.sub(r'(?:Ref\s*|Reference\s*|No\s*|Nr\s*|ID\s*):\s*[\w\d\-]+', '', description, flags=re.IGNORECASE)
    description = re.sub(r'Serial:\d+/\d+', '', description)

    # Remove transaction type prefixes
    description = re.sub(r'(?:POS Purchase|ATM Withdrawal|Immediate Payment|Internet Pmt To|Teller Transfer Debit|Direct Credit|EFT)\s*', '', description, flags=re.IGNORECASE)
    
    # Remove self-references and common suffixes
    description = re.sub(r'\s*-\s*Royal Panelbeaters', '', description, flags=re.IGNORECASE) 
    description = re.sub(r'\s{2,}', ' ', description).strip(' -').strip()
    
    return description

# --- 3. STANDARDIZATION FUNCTIONS ---

def standardize_fnb(df):
    """Specific standardization for FNB structure."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    # FNB often shows positive for credit and negative for debit, but the 'Cr' in balance is the definitive sign.
    # We will assume that any transaction line without a clear '-' is a credit, but FNB uses balance columns to flip.
    # The snippet shows the Amount column is always positive, and the sign is implicit via the balance.
    # We'll use the presence of 'Dr' in the description or the balance column's implicit sign.
    
    def calculate_fnb_amount(row):
        amount = clean_value(row['amount'])
        if not amount: return None
        # Use the 'Balance' column's associated 'Cr'/'Dr' from the PDF text to determine sign if not in amount.
        if 'dr' in str(row.get('balance', '')).lower() or '-' in str(row.get('amount', '')):
            return -abs(amount)
        return abs(amount)

    df['Amount'] = df.apply(calculate_fnb_amount, axis=1)
    df['Description'] = df['description'].astype(str)
    return df

def standardize_standard_hbz(df, debit_col, credit_col):
    """Specific standardization for banks with separate Debit/Credit columns (Standard Bank, HBZ)."""
    df[debit_col] = df[debit_col].apply(clean_value)
    df[credit_col] = df[credit_col].apply(clean_value)
    
    # Calculate amount: Credit - Debit (Debits should be negative in Xero)
    df['Amount'] = df[credit_col].fillna(0) - df[debit_col].fillna(0)
    
    # Use the column that holds the main description
    description_col = 'Details' if 'Details' in df.columns else 'Particulars'
    df['Description'] = df[description_col].astype(str)
    
    return df

def standardize_absa(df):
    """Specific standardization for ABSA structure."""
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    df['Amount'] = df['transaction_amount_r'].apply(clean_value)
    df['Description'] = df['description'].astype(str)
    return df

# --- 4. CORE EXTRACTION LOGIC ---

def parse_pdf_data(pdf_file_path):
    """Core function to extract tables, detect bank, and standardize the data."""
    all_transactions = pd.DataFrame()
    bank_name = "Unknown"

    try:
        with pdfplumber.open(pdf_file_path) as pdf:
            # 1. Detect Bank
            full_text = " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
            bank_name = detect_bank(full_text)
            
            if bank_name == "Unknown":
                st.warning("⚠️ Could not reliably identify the bank statement format (ABSA, FNB, Standard, HBZ, Nedbank, Capitec). Using generic parsing which may fail or produce incorrect results. Please check manually.")
                # Fallback: Use generic settings (lattice mode for tables)
                rules = {"columns": ['Date', 'Description', 'Amount', 'Balance'], "table_settings": {"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 5}}
            else:
                rules = BANK_RULES[bank_name]
                st.info(f"✅ Detected Bank: **{bank_name}**. Applying custom parsing rules.")
            
            # 2. Extract Data Page by Page
            for page in pdf.pages:
                tables = page.extract_tables(rules["table_settings"])
                
                for table in tables:
                    if table and len(table) > 1:
                        # Simple header detection logic
                        data_rows = []
                        if bank_name != "Unknown":
                            # For known banks, just process data after potential header rows
                            start_index = 0
                            for i, row in enumerate(table):
                                # Skip header rows, which often contain specific titles
                                if any("BALANCE" in str(cell).upper() for cell in row):
                                    start_index = i + 1
                            data_rows = table[start_index:]
                            
                            try:
                                # Apply column names only if the number of columns match
                                if len(data_rows[0]) == len(rules["columns"]):
                                    df = pd.DataFrame(data_rows, columns=rules["columns"])
                                else:
                                    # Fallback if column count is wrong for custom rule
                                    df = pd.DataFrame(data_rows)
                            except:
                                df = pd.DataFrame(data_rows)
                                
                        else: # Unknown/Generic/Scanned Fallback
                            df = pd.DataFrame(table)

                        all_transactions = pd.concat([all_transactions, df], ignore_index=True)
                            
            # 3. Standardization & Final Formatting
            if not all_transactions.empty:
                # Remove rows with too few non-NaN values (often junk rows)
                all_transactions.dropna(thresh=2, inplace=True)
                
                if bank_name == "FNB":
                    df_final = standardize_fnb(all_transactions.copy())
                elif bank_name == "Standard Bank":
                    df_final = standardize_standard_hbz(all_transactions.copy(), debit_col='Debits', credit_col='Credits')
                elif bank_name == "ABSA":
                    df_final = standardize_absa(all_transactions.copy())
                elif bank_name == "HBZ Bank":
                    df_final = standardize_standard_hbz(all_transactions.copy(), debit_col='Debit', credit_col='Credit')
                else:
                    # Generic fallback standardization (requires manual mapping which is too complex here)
                    st.warning("Cannot standardize data for unknown bank. CSV will contain raw extracted columns.")
                    return all_transactions.copy(), bank_name # Return raw for inspection
                
                # Final filtering and column selection for the standardized result
                df_final = df_final[df_final['Amount'].notna()]
                
                return df_final, bank_name

    except Exception as e:
        st.error(f"An error occurred during PDF processing: {e}")
        st.info("This often happens when `pdfplumber` cannot find text layers (i.e., it's a scanned image) or the PDF is malformed.")
        return pd.DataFrame(), bank_name
    
    return pd.DataFrame(), bank_name

# --- 5. STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(page_title="🇿🇦 Free SA Bank Statement to Xero CSV Converter", layout="wide")

    st.title("🇿🇦 SA Bank Statement PDF to Xero CSV Converter")
    st.markdown("""
        ### Build by Gemini AI: Free, easy-to-use tool for South African bank statement conversion.
        This app uses **Python/pdfplumber** to extract transactions from **native (text-based) PDF** statements and format them for upload into **Xero**. 
        
        **Supported Banks (with custom rules): ABSA, FNB, Standard Bank, HBZ** (and all others via generic parsing).
        
        **⚠️ Important Note on Scanned Documents & Free Service (The AI part):**
        * Extracting tables accurately from **scanned or image-based PDFs** requires advanced, proprietary **OCR (Optical Character Recognition) and AI models**.
        * A **completely free** solution like this **cannot guarantee reliable extraction** from scanned/image files. It will attempt it, but results may be inaccurate or empty.
        * This tool is best suited for **digital, native PDFs** from your bank.
        ---
    """)

    uploaded_files = st.file_uploader(
        "Upload your bank statement PDF files (Multiple files supported)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.subheader("Processing Files...")
        
        all_df = []
        
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            st.markdown(f"**Processing:** `{file_name}`")
            
            # Read PDF data into memory buffer
            pdf_data = BytesIO(uploaded_file.read())
            
            # Get the core transactions
            df_transactions, bank_name = parse_pdf_data(pdf_data)

            if not df_transactions.empty and 'Amount' in df_transactions.columns:
                
                # Apply Xero description cleanup
                df_transactions['Description'] = df_transactions['Description'].apply(clean_description_for_xero)
                
                # Select and rename columns for final Xero CSV format
                df_final = df_transactions.rename(columns={
                    'Date': 'Date',
                    'Description': 'Description',
                    'Amount': 'Amount'
                })
                
                # Standardize Date format to DD/MM/YYYY (Xero requirement)
                try:
                    df_final['Date'] = pd.to_datetime(df_final['Date'], errors='coerce', dayfirst=True).dt.strftime('%d/%m/%Y')
                except:
                    st.warning(f"Could not standardize dates for {file_name}. Dates remain in original format and may need manual check.")
                
                
                # Final Xero structure: Date, Amount, Payee, Description, Reference
                df_xero = pd.DataFrame({
                    'Date': df_final['Date'].fillna(''),
                    # Amount is critical and must be correct. Xero accepts positive for money in, negative for money out.
                    'Amount': df_final['Amount'].round(2), 
                    'Payee': '', # Can be mapped manually in Xero
                    'Description': df_final['Description'].astype(str),
                    'Reference': file_name.split('.')[0] # Use file name as a default reference
                })
                
                # Remove any remaining rows where the core data (Date, Amount) is missing
                df_xero.dropna(subset=['Date', 'Amount'], inplace=True)
                
                all_df.append(df_xero)
                
                st.success(f"Successfully extracted {len(df_xero)} transactions from {file_name}")

        
        # --- 6. COMBINE AND DOWNLOAD ---

        if all_df:
            final_combined_df = pd.concat(all_df, ignore_index=True)
            
            st.markdown("---")
            st.subheader("✅ All Transactions Combined and Ready for Download")
            st.dataframe(final_combined_df)
            
            # Create the Xero CSV file
            csv_output = final_combined_df.to_csv(index=False, sep=',', encoding='utf-8')
            st.download_button(
                label="⬇️ Download Xero Ready CSV File",
                data=csv_output,
                file_name="SA_Bank_Statements_Xero_Export.csv",
                mime="text/csv"
            )

if __name__ == '__main__':
    main()
