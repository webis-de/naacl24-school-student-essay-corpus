import pypdfium2 as pdfium
import re
import pandas as pd
import json

transcript_path = "./transcripts"
transcript_names = ["Transkript_01.pdf", "Transkript_02.pdf", "Transkript_03.pdf", "Transkript_04.pdf", "Transkript_05.pdf", "Transkript_06.pdf", "Transkript_07.pdf", "Transkript_08.pdf"]
table_path = "./transcripts/Datentabelle.xlsx"
annotations_path = "arg-school-corpus-annotations-v2.json"
write_to_path = "arg-school-corpus-created.json"

def main():
    #read transcripts
    transcript_dict = {}
    for transcript in transcript_names:
        path = transcript_path + "/" + transcript
        texts = [p.get_textpage().get_text_range() for p in pdfium.PdfDocument(path)]

        #remove title page
        texts = texts[1:] 

        #remove copyright notice
        c = "©"
        texts = [re.split(c, text)[0] for text in texts]
        texts = [text.strip() for text in texts]

        #merge texts that are split by page breaks
        i = 0
        while i < len(texts):
            if not "Messzeitpunkt" in texts[i]:
                texts[i-1:i+1] = [texts[i-1] + "\r\n" + texts[i]]
            else:
                i += 1

        #separate code from text
        for i in range(len(texts)):
            key = texts[i].split(",")
            if key[5].strip() == "Argumentation":
                key = key[0] + key[4]
                text = texts[i].split("\n",1)[1]
                text = text.replace("\r\n", "\n")
                text = text.replace("denn dann muss jeder nur\nein", "denn dann muss jeder nur ein")
                text = text.replace("geworfen,\nsondern ", "geworfen, sondern\n")
                text = text.replace("schuld ist,\naber den ", "schuld ist, aber den\n")
                text = text.replace("jemand anderes,\ndem ", "jemand anderes, dem\n")
                text = text.replace("hinter ihm, wenn\nden ", "hinter ihm, wenn den\n")
                text = text.replace("dass du es\nwarst ", "dass du es warst\n")
                text = text.replace("gesagt\nhat.", "gesagt hat.")
                text = text.replace("will ich\nIhnen ", "will ich Ihnen\n")
                text = text.replace("besser zu\ngestalten, ", "besser zu gestalten,\n")
                text = text.replace("ausgetobt und sind im\nUnterricht", "ausgetobt und sind im Unterricht")
                text = text.replace("verpetzt werden würde,\nwürde", "verpetzt werden würde, würde")
                text = text.replace("jemand anderes,\ndem", "jemand anderes, dem")
                text = re.sub(r"￾", "", text)
                text = text.strip()
                transcript_dict.update({key:text})

    #read metadata
    table = pd.read_excel("transcripts/Datentabelle.xlsx")
    #open annotations
    with open(annotations_path, 'r', encoding="utf-8") as f:
            corpus = json.load(f)

    #fill corpus gaps
    for instance in corpus:
            instance["text"] = transcript_dict[instance["fdlex_id"]+ " " + instance["mzp"]]
            for segment in instance["macro_l1"]:
                segment["text"] = instance["text"][segment["start"]: (segment["end"])]
            for segment in instance["macro_l2"]:
                segment["text"] = instance["text"][segment["start"]: (segment["end"])]
            for segment in instance["micro_l1"]:
                segment["text"] = instance["text"][segment["start"]: (segment["end"])]
            for segment in instance["micro_l2"]:
                segment["text"] = instance["text"][segment["start"]: (segment["end"])]
            tablerow = table.loc[table["Code"] == instance["fdlex_id"]]
            instance["group"] = tablerow["Gruppe"].item()
            instance["grade"] = tablerow["Klassenstufe"].item()
            instance["school"] = tablerow["Schulform"].item()
            instance["age"] = tablerow["Alter in Monaten"].item()
            instance["gender"] = tablerow["Geschlecht"].item()
            instance["language"] = tablerow["Sprachbiographie"].item()
            if pd.isna(tablerow["Deutschnote"].item()):
                instance["german_grade"] = 0.0
            else:
                instance["german_grade"] = tablerow["Deutschnote"].item()

        
    # write new completed corpus
    with open(write_to_path, 'w', encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
