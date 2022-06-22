# Dementia Cognitive Test

## Introduzione
Per demenza si intende un termine generale che indica un declino delle capacità mentali molta gravi da interferire con la vita quotidiana. La perdita di memoria ne è un esempio.

La demenza non è una malattia specifica, ma un termine generale che descrive un grupo di sintomi associata ad un declino della memoria o di altre capacità mentali abbastanza grave da ridurre la capacità di una persona di svolgere le attività quotidiane.


### Informazioni sul dataset utilizzato

Il dataset utilizzato per l'addestramento del sistema è tratto dal sito www.kaggle.com ed è visualizzabile tramite il seguente link

https://www.kaggle.com/code/obrienmitch94/alzheimer-s-analysis/report

Questo dataset si basa su una raccolta longitudinale di 150 soggetti di età compresa tra 60 e 96 anni.

Ciascun soggetto del database è stato sottoposto a risonanza magnetica pesata' in T1 in due o più visite, a distanza di almeno un anno, per un totale di 373 scansioni di imaging.

Sono presenti 9 feature binarie che rappresentano dati vari, e 5 feature di categoria.



|Feature                          |Description                         |
|-------------------------------|-----------------------------|
|ID        |Identificativo          |
Group     |Stato della demenza      |
|Visit|Numero della vita|
|M/F|Sesso della paziente|
|Hand| Mano predominante|
|Age|Età del paziente|
|Educ|Anni di educazione|
|SES|Stato economico|
|MMSE|Esame ridotto dello stato mentale|
|CDR|Valutazione della demenza clinica|
|eTIV|Volume Intranico Totale|
|nWBV|Volume Normalizzato dell'intero cervello|
|ASF|Atlas Scaling Factor|

## Esecuzione

Installare le dipendenze: 

    pip install -r requirements.txt

Eseguire il programma

    python main.py

