import json
import re 
import pandas as pd


def main():
    #load corpus with no texts and datatable
    with open('../data/arg-school-corpus-annotations.json', 'r') as f:
        corpus = json.load(f)
    with open('../data/Datentabelle.csv', 'r') as f:
        table = pd.read_csv(f,delimiter=";")
    transcript_dict = {}

    #create dict from transcript files with format {(FDLEX_ID) (MZP): (TEXT)}
    for filename in ['Transkript_01.txt', 'Transkript_02.txt', 'Transkript_03.txt', 'Transkript_04.txt']:
        with open('../data/' + filename, 'r') as f:
            transkript = f.read()
        matches = re.findall(".*,.*,.*,.*,.*,.*,[^\u00a9]*", transkript)
      
        for match in matches:
            
            key = match.split(",")
            key = key[0] + key[4]
            text = match.split("\n",1)[1]
            transcript_dict.update({key:text})
            
    # fill corpus gaps    
    for instance in corpus:
        instance["text"] = transcript_dict[instance["fdlex_id"]+ " " + instance["mzp"]][:-1]
        for segment in instance["macro_l1"]:
            segment["text"] = instance["text"][segment["start"]: (segment["end"])]
        for segment in instance["macro_l2"]:
            segment["text"] = instance["text"][segment["start"]: (segment["end"])]
        for segment in instance["micro_l1"]:
            segment["text"] = instance["text"][segment["start"]: (segment["end"])]
        for segment in instance["micro_l2"]:
            segment["text"] = instance["text"][segment["start"]: (segment["end"])]
        tablerow = table[table["Code"] == instance["fdlex_id"]]
        instance["group"] = tablerow["Gruppe"].item()
        instance["grade"] = tablerow["Klassenstufe"].item()
        instance["school"] = tablerow["Schulform"].item()
        instance["age"] = tablerow["Alter in Monaten"].item()
        instance["gender"] = tablerow["Geschlecht"].item()
        instance["language"] = tablerow["Sprachbiographie"].item()
        instance["german_grade"] = tablerow["Deutschnote"].item()

    # write new completed corpus
    with open('../data/arg-school-corpus-created.json', 'w') as f:
        json.dump(corpus, f)
   


if __name__ == "__main__":
    main()
