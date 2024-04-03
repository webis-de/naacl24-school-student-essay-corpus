# macro l1
id2label_macro_l1 = {
    0: "O",
    1: "B-Einleitung",
    2: "I-Einleitung",
    3: "B-Hauptteil",
    4: "I-Hauptteil",
    5: "B-Konklusion",
    6: "I-Konklusion",
}
label2id_macro_l1 = {
    "O": 0,
    "B-Einleitung": 1,
    "I-Einleitung": 2,
    "B-Hauptteil": 3,
    "I-Hauptteil": 4,
    "B-Konklusion": 5,
    "I-Konklusion": 6,
}

# macro l2
id2label_macro_l2 = {
    0: "O",
    1: "B-Argument",
    2: "I-Argument",
    3: "B-Gegenargument",
    4: "I-Gegenargument",
}
label2id_macro_l2 = {
    "O": 0,
    "B-Argument": 1,
    "I-Argument": 2,
    "B-Gegenargument": 3,
    "I-Gegenargument": 4,
}

# micro l1
id2label_micro_l1 = {
    0: 'O',
    1: 'B-Thema',
    2: 'I-Thema',
    3: 'B-These',
    4: 'I-These',
    5: 'B-Gegenthese',
    6: 'I-Gegenthese',
    7: 'B-Modifizierte-These',
    8: 'I-Modifizierte-These',
    9: 'B-Claim',
    10: 'I-Claim',
    11: 'B-Premise',
    12: 'I-Premise',
}
label2id_micro_l1 = {
    'O': 0,
    'B-Thema': 1,
    'I-Thema': 2,
    'B-These': 3,
    'I-These': 4,
    'B-Gegenthese': 5,
    'I-Gegenthese': 6,
    'B-Modifizierte-These': 7,
    'I-Modifizierte-These': 8,
    'B-Claim': 9,
    'I-Claim': 10,
    'B-Premise': 11,
    'I-Premise': 12,
}


# micro l2
id2label_micro_l2 = {
    0: 'O',
    1: 'B-Positionieren',
    2: 'I-Positionieren',
    3: 'B-Beschreiben',
    4: 'I-Beschreiben',
    5: 'B-Exemplifizieren',
    6: 'I-Exemplifizieren',
    7: 'B-Begründen',
    8: 'I-Begründen',
    9: 'B-Konzedieren',
    10: 'I-Konzedieren',
    11: 'B-Referieren',
    12: 'I-Referieren',
    13: 'B-Abwägen',
    14: 'I-Abwägen',
    15: 'B-Auffordern',
    16: 'I-Auffordern',
    17: 'B-Einschränken',
    18: 'I-Einschränken',
    19: 'B-Schlussfolgern',
    20: 'I-Schlussfolgern',
}
label2id_micro_l2 = {
    'O': 0,
    'B-Positionieren': 1,
    'I-Positionieren': 2,
    'B-Beschreiben': 3,
    'I-Beschreiben': 4,
    'B-Exemplifizieren': 5,
    'I-Exemplifizieren': 6,
    'B-Begründen': 7,
    'I-Begründen': 8,
    'B-Konzedieren': 9,
    'I-Konzedieren': 10,
    'B-Referieren': 11,
    'I-Referieren': 12,
    'B-Abwägen': 13,
    'I-Abwägen': 14,
    'B-Auffordern': 15,
    'I-Auffordern': 16,
    'B-Einschränken': 17,
    'I-Einschränken': 18,
    'B-Schlussfolgern': 19,
    'I-Schlussfolgern': 20,
}

# quality
id2label_quality = {
    0: 1,
    1: 1.5, 
    2: 2, 
    3: 2.5,
    4: 3,
    5: 3.5, 
    6: 4
}
label2id_quality = {
    1.0: 0,
    1.5: 1,
    2.0: 2,
    2.5: 3,
    3.0: 4,
    3.5: 5,
    4.0: 6,
}

quality_dimensions = [
    'textfunktion', 
    'inhaltliche_ausgestaltung', 
    'textstruktur', 
    'sprachliche_ausgestaltung',
    'gesamteindruck'
]
