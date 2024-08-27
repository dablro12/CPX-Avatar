import pdfplumber
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import os

def pdf2png(pdf_path, output_folder):
    # PDF를 이미지로 변환
    images = convert_from_path(pdf_path)
    
    # 저장 디렉토리 설정
    save_dir = os.path.join(output_folder, os.path.basename(pdf_path).replace('.pdf', ''))
    os.makedirs(save_dir, exist_ok=True)
    
    # 각 페이지를 이미지로 저장
    for page_number, image in enumerate(images):
        # 이미지 저장
        image.save(f"{save_dir}/page_{page_number + 1}.png")
        print(f"Saved: {save_dir}/page_{page_number + 1}.png")

# Example 
# # PDF 경로와 출력 폴더 설정
# pdf_path = '../data/cough_table.pdf'
# output_folder = "/home/eiden/eiden/LLM/CPX-Avatar/data/out"

# # 함수 실행
# pdf2png(pdf_path=pdf_path, output_folder=output_folder)
