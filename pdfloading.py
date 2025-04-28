import os
datasets_folder = 'C:\Users\47770057\Desktop\NBot\Dataset'

pdf_files = [
    os.path.join(datasets_folder, f)
    for f in os.listdir(datasets_folder)
    if f.lower().endswith(".pdf")
]


print(pdf_files)
