import os
from fpdf import FPDF

class PDFHandler:
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def save_text_to_pdf(self, text, file_name):
        pdf = FPDF()

        # Add a page
        pdf.add_page()

        # Set font
        font_path = os.path.join(os.path.dirname(__file__), 'DejaVuSansCondensed.ttf')
        pdf.add_font('DejaVu', '', font_path, uni=True)
        pdf.set_font('DejaVu', size=12)

        # Calculate maximum cell width
        max_cell_width = pdf.w - 2 * pdf.l_margin
        text = text.replace('\t', '')
        text = text.replace("\\", "")
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        print(paragraphs)

        for paragraph in paragraphs:
            lines = paragraph.split('\n')
            for line in lines:
                pdf.multi_cell(max_cell_width, 10, txt=line)
            pdf.ln(10)  # add an extra line between paragraphs

        # Define file path
        file_path = os.path.join(self.dir_path, f'{file_name}.pdf')

        # Save the pdf with name .pdf
        pdf.output(file_path)

        # Return the file path
        return file_path