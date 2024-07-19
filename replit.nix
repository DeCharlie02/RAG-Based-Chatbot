{ pkgs }: {
deps = [
pkgs.python39Full
pkgs.python39Packages.pip
pkgs.tesseract
pkgs.libjpeg
pkgs.zlib
];

postBuild = ''
pip install streamlit PyPDF2 pillow pytesseract faiss-cpu numpy torch transformers openai langchain langsmith
'';
}