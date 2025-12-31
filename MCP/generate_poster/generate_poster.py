import sys
import os
import logging
from PIL import Image
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from pypdf import PdfReader
import requests
import json
import matplotlib.pyplot as plt
import os

from utils import create_vertical_poster

# Set up logging to a file instead of redirecting stdout
logging.basicConfig(
    filename='.debug_mcp.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PosterGenerator:
    def __init__(self):
        load_dotenv()
        self.mcp = FastMCP("generate_poster")
        logger.info("MCP Server initialized")
        self._register_tools()
    
    def _register_tools(self):
        """Register MCP tools for poster generation."""
        
        @self.mcp.tool()
        def create_poster_from_pdf_auto(
            folder_name: str,
            document_name: str,
        ) -> str:
            """Complete workflow: Load PDF, extract text/images, create poster."""
            logger.info(f"create_poster_from_pdf_auto called: {folder_name}/{document_name}")
            try:
                # Step 1: Load PDF text
                file_path = Path(folder_name) / document_name
                if not file_path.exists():
                    return f"File not found: {file_path}"
                
                logger.info("Step 1: Loading PDF text...")
                reader = PdfReader(file_path)
                all_text = []
                for i, page in enumerate(reader.pages):
                    text = page.extract_text().replace('\x00', '').strip()
                    if text:
                        all_text.append(text)
                
                # FULL PAPER TEXT (for understanding key points):
                full_text = " ".join(all_text)
                logger.info(f"Extracted {len(full_text)} characters of text")
                
                # Step 1.5: Create structured summary using qwen
                logger.info("Step 1.5: Creating structured summary...")
                summary_prompt = f"""Based on this text, create a structured summary with the following mandatory fields:
                                1. Paper title
                                2. Authors: List of author names.
                                3. Affilications: Authors' corresponding institutes
                                4. Venue of the publication (if it's from arXiv, using arXiv prepreint with number)
                                5. Max 5 key bullet points (each max 15 words)

                                Text:
                                {full_text}

                                Respond ONLY with valid JSON in this exact format:
                                {{
                                    "title": "Paper title",
                                    "authors": ["Author 1", "Author 2", "Author 3"],
                                    "affilications": ["Institute 1","Institute 2", "Institute 3"],
                                    "venue": "Venue",
                                    "points": [
                                        "First key point",
                                        "Second key point",
                                        "Third key point",
                                    ]
                                }}"""

                try:
                    summary_response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "qwen3:30b",
                            "prompt": summary_prompt,
                            "stream": False,
                            "options": {"temperature": 0.6}
                        },
                        timeout=300
                    )
                    
                    if summary_response.status_code == 200:
                        summary_result = summary_response.json()
                        summary_text = summary_result.get('response', '')
                        
                        # Extract JSON
                        summary_text = summary_text.replace('```json', '').replace('```', '').strip()
                        start = summary_text.find('{')
                        end = summary_text.rfind('}') + 1
                        
                        if start != -1 and end > start:
                            summary_data = json.loads(summary_text[start:end])
                        else:
                            raise ValueError("No JSON found")
                        
                        authors = summary_data.get('authors', [])
                        if len(authors) > 3:
                            summary_data['authors'] = authors[:3] + ["et al."]
                            logger.info(f"Truncated authors from {len(authors)} to 3 + et al.")
                    else:
                        raise ValueError("Summary request failed")
                        
                except Exception as summary_error:
                    logger.warning(f"Summary generation failed: {summary_error}")
                    # Fallback summary
                    summary_data = {
                        "title": document_name.replace('.pdf', '').replace('_', ' ').title(),
                        "points": [full_text[:150] + "..."]
                    }
                
                logger.info(f"Summary data: {summary_data}")
                
                # Step 2: Extract first image from PDF
                logger.info("Extracting images from PDF...")
                extracted_any = False
                
                for page_idx, page in enumerate(reader.pages):
                    if extracted_any:
                        break
                        
                    if '/Resources' in page and '/XObject' in page['/Resources']:
                        xobjects = page['/Resources']['/XObject'].get_object()
                        
                        for obj_name in xobjects:
                            if extracted_any:
                                break
                                
                            obj = xobjects[obj_name]
                            
                            if obj['/Subtype'] == '/Image':
                                try:
                                    size = (obj['/Width'], obj['/Height'])
                                    data = obj.get_data()
                                    mode = 'RGB'
                                    
                                    if '/ColorSpace' in obj:
                                        color_space = obj['/ColorSpace']
                                        if color_space == '/DeviceGray':
                                            mode = 'L'
                                        elif color_space == '/DeviceCMYK':
                                            mode = 'CMYK'
                                    
                                    poster_img = Image.frombytes(mode, size, data)
                                    
                                    if poster_img.mode == 'CMYK':
                                        poster_img = poster_img.convert('RGB')
                                    
                                    logger.info(f"Extracted image from page {page_idx + 1}")
                                    extracted_any = True
                                    break
                                    
                                except Exception as img_error:
                                    logger.warning(f"Failed to extract image: {img_error}")
                                    continue
                
                # Step 3: Create poster with structured layout
                create_vertical_poster(summary_data, poster_img)
                return "Success creating poster!"
                
            except Exception as e:
                error_msg = f"Error in auto poster creation: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return error_msg
    
    def run(self):
        """Start the MCP server."""
        logger.info("Starting MCP Server...")
        self.mcp.run(transport="stdio")

if __name__ == "__main__":
    logger.info("=== Poster Generator Starting ===")
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    analyzer = PosterGenerator()
    analyzer.run()