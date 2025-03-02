from itertools import product

def extract_time(s: str) -> list:
    # Finden der Positionen der relevanten Teile des Strings
    start=len(s)-22
    y_idx = s.find('y',start)
    m_idx = s.find('m',start)
    d_idx = s.find('d',start)
    h_idx = s.find('h',start)
    min_idx = s.find('m', h_idx)

    # Extrahieren der Zahlen als Substrings und Umwandeln in int
    year = int(s[y_idx-4:y_idx])
    month = int(s[y_idx+1:m_idx])
    day = int(s[m_idx+1:d_idx])
    hour = int(s[d_idx+2:h_idx])
    minute = int(s[h_idx+1:min_idx])

    return (year, month, day, hour, minute)


# Funktion, um Werte anzupassen (größer als 1000 -> 'xK')
import pandas as pd
def format_thousand_annotation(val):
    #print('a',val,type(val))
    if pd.isna(val) or pd.isnull(val) or type(val) == str:
        return ""  # Leere Zellen für NaN-Werte
    elif val >= 1000:
        return f"{val / 1000:.1f}K"  # Zeigt nur eine Dezimalstelle
    else:
        return f"{val:.0f}"  # Ganze Zahl für Werte unter 1000

import os

# Funktion, um alle Ordner in einem bestimmten Verzeichnis zu finden und ihre Namen zu speichern
def find_folders(directory):
    folders = []
    for item in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, item)):
            folders.append(item)
    return folders

# Funktion, um alle Dateien in einem bestimmten Verzeichnis zu finden und ihre Namen zu speichern
def find_files(directory):
    files = []
    for item in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, item)) and item.endswith('zvi'):
            files.append(item)
    return files


# Funktion zur Bestimmung der Replikate
def find_replicates(dataframe_dict: dict) -> dict:
    '''

        Finds the replicates of a dictionary of pd.Dataframes

    :param dataframe_dict: dictionary of dataframes, where the key is the name of the dataframe
    :return: dictionary of the form {(('HepG2', 20000.0), ('Hep3B', 20000.0)): ['B2', 'C2'], ...}
    '''
    replikate = {}
    keys = [key for key in dataframe_dict.keys()]

    # Alle möglichen Positionen (Zeile, Spalte)
    positionen = list(product(dataframe_dict[keys[0]].index, dataframe_dict[keys[0]].columns))

    # Schleife über alle Positionen
    for pos in positionen:
        # Bestimme Werte für jede Position für jeden df
        werte = {name: df.at[pos[0], pos[1]] for name, df in dataframe_dict.items()}
        # Konvertiere NaN-Werte zu None, um sie später zu ignorieren
        werte = {k: (v if pd.notna(v) else None) for k, v in werte.items()}

        # Überspringen, wenn einer der Werte None ist (keine vollständigen Replikate)
        if all(value is None for value in werte.values()):
            continue

        # Spezifikation als Tupel zur Verwendung als Dictionary-Schlüssel
        werte_tupel = tuple(werte.items())

        # Konvertiere Position in das formatierte Format (z.B. "B2")
        pos_label = f"{pos[0]}{pos[1]}"

        # Füge Position zur Liste der jeweiligen Replikat-Spezifikation hinzu
        if werte_tupel not in replikate:
            replikate[werte_tupel] = []
        replikate[werte_tupel].append(pos_label)

    return replikate