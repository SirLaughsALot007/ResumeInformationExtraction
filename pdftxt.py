import fitz
import os 
def get_text(filepath: str) -> str:
    with fitz.open(filepath) as doc:
        text = ""
        for page in doc:
            text += page.getText().strip()
        return text
if __name__ == '__main__':
    results = []
    paths = r'D:\code\wyl\resumes\pdf_resumes'
    print(paths)
    resumes = os.listdir(paths)
    fp = open('test_onepiece.txt','w',encoding='utf-8')
    for i in range(10):
        path = os.path.join(paths,resumes[i])
        str = get_text(path)
        str = str.replace(' ','')
        str = '\tO\n'.join(str)
        results.append(str)
        print(results[i])
        fp.write(str)
        fp.write('\to\n')    
    fp.close()