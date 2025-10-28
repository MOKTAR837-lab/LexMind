import os
import base64
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from PIL import Image
import pytesseract
import pdf2image
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np

# Configure Tesseract path (Docker)
if os.path.exists('/usr/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
else:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class MultiModalProcessor:
    def __init__(self):
        print("🎨 Initialisation Multi-Modal Processor...")
        
        # CLIP Model pour embeddings visuels
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model.to(self.device)
            print(f"✅ CLIP model charge (device: {self.device})")
        except Exception as e:
            print(f"⚠️ CLIP non disponible: {e}")
            self.clip_model = None
        
        # OCR Config
        self.ocr_langs = 'fra+eng'
        
        print("✅ Multi-Modal Processor pret\n")
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract texte depuis image avec OCR"""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang=self.ocr_langs)
            return text.strip()
        except Exception as e:
            print(f"❌ OCR error: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path: str, max_pages: int = 50) -> List[Dict]:
        """Extract texte depuis PDF"""
        results = []
        
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(min(len(pdf_reader.pages), max_pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    
                    if len(text.strip()) < 50:
                        try:
                            images = pdf2image.convert_from_path(
                                pdf_path, 
                                first_page=page_num+1, 
                                last_page=page_num+1,
                                dpi=300
                            )
                            if images:
                                text = pytesseract.image_to_string(images[0], lang=self.ocr_langs)
                        except:
                            pass
                    
                    if text.strip():
                        results.append({
                            'page': page_num + 1,
                            'text': text.strip(),
                            'method': 'native' if len(text.strip()) >= 50 else 'ocr'
                        })
        
        except Exception as e:
            print(f"⚠️ PDF extraction error: {e}, trying full OCR...")
            try:
                images = pdf2image.convert_from_path(pdf_path, dpi=300)
                for i, image in enumerate(images[:max_pages]):
                    text = pytesseract.image_to_string(image, lang=self.ocr_langs)
                    if text.strip():
                        results.append({
                            'page': i + 1,
                            'text': text.strip(),
                            'method': 'ocr'
                        })
            except Exception as e2:
                print(f"❌ Full OCR failed: {e2}")
        
        return results
    
    def get_image_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Generate embedding visuel avec CLIP"""
        if not self.clip_model:
            return None
        
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            embedding = image_features / image_features.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"❌ Image embedding error: {e}")
            return None
    
    def get_text_embedding_clip(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding texte avec CLIP"""
        if not self.clip_model:
            return None
        
        try:
            inputs = self.clip_processor(text=text, return_tensors="pt", padding=True).to(self.device)
            
            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
            
            embedding = text_features / text_features.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy().flatten()
        
        except Exception as e:
            print(f"❌ Text embedding CLIP error: {e}")
            return None
    
    def process_document(self, file_path: str, doc_type: str = None) -> Dict:
        """Process document multi-modal complet"""
        
        if doc_type is None:
            ext = os.path.splitext(file_path)[1].lower()
            doc_type = ext[1:]
        
        result = {
            'file_path': file_path,
            'type': doc_type,
            'text': '',
            'pages': [],
            'embeddings': {}
        }
        
        if doc_type in ['png', 'jpg', 'jpeg', 'tiff', 'bmp']:
            result['text'] = self.extract_text_from_image(file_path)
            image_emb = self.get_image_embedding(file_path)
            if image_emb is not None:
                result['embeddings']['visual'] = image_emb.tolist()
        
        elif doc_type == 'pdf':
            pages = self.extract_text_from_pdf(file_path)
            result['pages'] = pages
            result['text'] = '\n\n'.join([p['text'] for p in pages])
        
        elif doc_type in ['txt', 'md', 'docx']:
            try:
                if doc_type == 'docx':
                    import docx
                    doc = docx.Document(file_path)
                    result['text'] = '\n'.join([p.text for p in doc.paragraphs])
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        result['text'] = f.read()
            except Exception as e:
                print(f"❌ Text extraction error: {e}")
        
        return result
