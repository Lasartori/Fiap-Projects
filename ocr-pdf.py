from paddleocr import PaddleOCR

# Inicializa o OCR com o modelo correto
ocr = PaddleOCR(use_textline_orientation=True, lang='en', enable_mkldnn = False)  # Use 'en' em vez de 'eng'

pdf_path = 'gallery/guide-ny.pdf'

def extract_text_from_pdf(pdf_path):
    extract_ocr = ocr.ocr(pdf_path)
    result = ''
    for line in extract_ocr:
        for word in line:
            result += word[1][0] + ' '  # Adicionei um espaço entre palavras
        result += '\n'
    return result

ocr_text = extract_text_from_pdf(pdf_path)
print(ocr_text)

