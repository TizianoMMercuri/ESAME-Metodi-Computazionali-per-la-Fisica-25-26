import funz as f #modulo che contiene le funzioni costruite
import argparse


#definizione del codice argparse per separare le due parti principali del progetto 


def argomenti_parse():
    parser = argparse.ArgumentParser(description="Spiegazione comandi.", usage="python3 tramonti.py  --opzione")
    parser.add_argument("-I", "--Interattivo", action="store_true", help="Esegue in modalità interattiva")#aggiungo la prima parte
    parser.add_argument("-O", "--Ozono", action="store_true", help="Consente la visualizzazione sullo studio dell'Ozono")#aggiungo la seconda parte
    return parser.parse_args()
def main():
    args=argomenti_parse()
    if args.Interattivo:
        f.menu_interattivo()
    if args.Ozono:
        f.Stud_O3()
    else:
        print("Uscita dal programma...")
if __name__== "__main__":
    main()
