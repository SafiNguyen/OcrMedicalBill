# OCR and Keyword Extraction for Vietnamese Medical Prescriptions

Desktop application for automatic text extraction and classification from Vietnamese medical prescription images using Tesseract OCR and KeyBERT.

## Requirements

- Python 3.8+
- Tesseract OCR 4.0+
- Vietnamese language data (vie.traineddata)

## Installation

1. Install Tesseract OCR:

   - Windows: https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: `sudo apt-get install tesseract-ocr tesseract-ocr-vie`
   - macOS: `brew install tesseract tesseract-lang`

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Update Tesseract path in `config.yaml` if needed

## Usage

```bash
python main.py
```

## Configuration

Edit `config.yaml` to customize:

- Tesseract path
- OCR parameters
- Keyword classification rules
- Model settings

## Performance

- Processing time: 5-10 seconds per document
- Optimized for printed text
- Best results with clear, well-lit images

## Limitations

- Limited handwriting recognition
- Optimized for Vietnamese language only
- Complex table structures may require verification

## License

MIT License

## Acknowledgments

- Tesseract OCR
- KeyBERT
- OpenCV
- Sentence Transformers
