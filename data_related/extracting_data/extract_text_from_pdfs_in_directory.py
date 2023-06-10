import os
import glob
from pdfminer.high_level import extract_text

# Set the directory and text file paths
directory = r'__DIRECTORY__'
output_file_path = os.path.join(directory, '../extract_text_from_pdfs_in_directory/combined_text.txt')

pdf_files = glob.glob(os.path.join(directory, '*.pdf'))

with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for pdf_file in pdf_files:
        # Extract text from the PDF file
        text = extract_text(pdf_file)

        # Append the extracted text to the output text file
        output_file.write(text)
        output_file.write("\n\n-----\n\n")

print(f"Text from all PDF files saved to: {output_file_path}")
