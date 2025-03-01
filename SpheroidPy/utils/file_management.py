import os
import re
from datetime import datetime, timedelta


def parse_filename_or_dirname(name) -> tuple:
    """
    Diese Funktion extrahiert den Index und den Zeitpunkt aus einem Ordner- oder Dateinamen.
    Es werden verschiedene Datums- und Zeitformate berücksichtigt und in einheitliches Format gebracht.
    """
    # Muster für Index (z.B. A1, B3, H12)
    index_pattern = r'(_[A-H][1-9][0-2]?)'

    # Verschiedene Datums- und Zeitmuster
    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD (2023-10-05)
        r'(\d{2}-\d{2}-\d{4})',  # DD-MM-YYYY (05-10-2023)
        r'(\d{4}y\d{2}m\d{2}d)',  # YYYYyMMmDDd (2024y09m14d)
    ]

    time_patterns = [
        r'(\d{2}:\d{2}:\d{2})',  # HH:mm:ss (15:30:45)
        r'(\d{2}:\d{2})',  # HH:mm (15:30)
        r'(\d{2}h\d{2}m)',  # HHhMMm (16h00m)
    ]

    relative_time_patterns = [
        r'(\d+)([hmsd])',  # Relative Zeitangaben (z.B. 15h, 2d)
    ]

    index = None
    timestamp = None

    # Zuerst nach dem Index suchen
    index_match = re.search(index_pattern, name)
    if index_match:
        index = index_match.group(0)
        index = index[1:]

    # Durch alle Date-Patterns iterieren und das erste gefundene verwenden
    for pattern in date_patterns:
        date_match = re.search(pattern, name)
        if date_match:
            date_str = date_match.group(0)
            # Einheitsformat (YYYY-MM-DD)
            if re.match(r'\d{2}-\d{2}-\d{4}', date_str):  # DD-MM-YYYY
                date_str = datetime.strptime(date_str, '%d-%m-%Y').strftime('%Y-%m-%d')
            elif re.match(r'\d{4}y\d{2}m\d{2}d', date_str):  # YYYYyMMmDDd
                date_str = datetime.strptime(date_str, '%Yy%mm%dd').strftime('%Y-%m-%d')
            # Ansonsten: YYYY-MM-DD ist schon korrekt
            timestamp = date_str
            break

    # Wenn ein Datum gefunden wurde, nach Zeitangaben suchen
    if timestamp:
        for pattern in time_patterns:
            time_match = re.search(pattern, name)
            if time_match:
                time_str = time_match.group(0)
                # Standardformat (HH:mm:ss)
                if re.match(r'\d{2}h\d{2}m', time_str):  # HHhMMm
                    time_str = datetime.strptime(time_str, '%Hh%Mm').strftime('%H:%M:%S')
                timestamp = f"{timestamp} {time_str}"
                break

    # Wenn weder Datum noch Zeitangabe im Format HH:mm gefunden wurde, nach relativen Zeitangaben suchen
    if not timestamp:
        for pattern in relative_time_patterns:
            rel_time_match = re.search(pattern, name)
            if rel_time_match:
                time_value = int(rel_time_match.group(1))
                time_unit = rel_time_match.group(2)

                if time_unit == 'h':  # Stunden
                    relative_time = timedelta(hours=time_value)
                elif time_unit == 'd':  # Tage
                    relative_time = timedelta(days=time_value)
                elif time_unit == 'm':  # Minuten
                    relative_time = timedelta(minutes=time_value)
                elif time_unit == 's':  # Sekunden
                    relative_time = timedelta(seconds=time_value)

                # Zeitangabe in lesbarer Form (z.B. "in 2 Tagen")
                future_time = datetime.now() + relative_time
                timestamp = future_time.strftime('%Y-%m-%d %H:%M:%S')
                break

    return index, timestamp


def print_directory_tree(base_dir, indent_level=0, filter_strs=None) -> None:
    """
    Diese Funktion druckt die Verzeichnisstruktur rekursiv als Baum.
    Optional können Filterstrings verwendet werden, um nur bestimmte Dateien/Ordner anzuzeigen.
    """
    # Fügt für jede Ebene eine Einrückung hinzu
    indent = '    ' * indent_level
    items = os.listdir(base_dir)

    for item in items:
        item_path = os.path.join(base_dir, item)

        # Optionaler Filter
        if matches_filter(item, filter_strs):
            # Wenn das Item ein Verzeichnis ist, rekursiv weitergehen
            if os.path.isdir(item_path):
                print(f"{indent}📁 {item}/")  # Ordner mit Symbol anzeigen
                print_directory_tree(item_path, indent_level + 1, filter_strs)  # Rekursion für den Ordner
            else:
                print(f"{indent}📄 {item}")  # Datei mit Symbol anzeigen


def matches_filter(name, filter_strs) -> bool:
    """
    Überprüft, ob der gegebene Dateiname/Ordnername mit dem optionalen Filter übereinstimmt.
    """
    # Wenn kein Filter angegeben wurde, wird immer True zurückgegeben
    if not filter_strs:
        return True

    # Wenn der Filter ein einzelner String ist, in den Namen suchen
    if isinstance(filter_strs, str):
        return filter_strs in name

    # Wenn der Filter eine Liste von Strings ist, müssen alle darin vorkommen
    if isinstance(filter_strs, list):
        return all(f in name for f in filter_strs)

    return False


def collect_data(base_dir: str, filter_strs = None) -> tuple:
    """
    Diese Funktion durchläuft rekursiv die Ordnerstruktur und sammelt Daten für jeden Index,
    wobei jeder Index als Schlüssel und die Zeitpunkte als Unterschlüssel verwendet werden.
    Zusätzlich werden alle gefundenen Zeitpunkte gesammelt und sortiert.
    """
    data = {}
    timestamps_list = []  # Liste, um alle Zeitpunkte zu sammeln

    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if matches_filter(dir_name, filter_strs):
                index, timestamp = parse_filename_or_dirname(dir_name)
                if index and timestamp:
                    if index not in data:
                        data[index] = {}
                    if timestamp not in data[index]:
                        data[index][timestamp] = []
                    data[index][timestamp].append(os.path.join(root, dir_name))

                    # Zeitpunkt in Liste speichern, falls noch nicht vorhanden
                    if timestamp not in timestamps_list:
                        timestamps_list.append(timestamp)

        for file_name in files:
            if matches_filter(file_name, filter_strs):
                #print(f"Including file: {file_name}")
                index, timestamp = parse_filename_or_dirname(file_name)
                #print(index, timestamp)
                if index and timestamp:
                    if index not in data:
                        data[index] = {}
                    data[index][timestamp] = os.path.join(root, file_name)

                    # Zeitpunkt in Liste speichern, falls noch nicht vorhanden
                    if timestamp not in timestamps_list:
                        timestamps_list.append(timestamp)

    # Sortiere die Liste der Zeitpunkte
    timestamps_list.sort(key=lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

    return data, timestamps_list

# Define a function to print direct subgroups of a given group
def direct_subgroups(group_name, hdf) -> list | None:
    if group_name in hdf:
        subgroups_array = []
        for subgroup in hdf[group_name].keys():
            subgroups_array.append(subgroup)
    else:
        subgroups_array = None
    return subgroups_array


if __name__ == '__main__':
    from pathlib import Path

    # Beispielaufruf
    base_directory = Path(
        '/Users/cedric/Desktop/Labor/Data/Incucyte/SpheroidProliferation/Hep3B')  # '/path/to/your/data'  # Hier den Pfad zu den Daten anpassen
    filter_strings = ['PhaseContrast', 'raw']  # Beispielhafter Filter: Dateien/Ordner müssen 'example' und 'test' enthalten
    result = collect_data(base_directory, filter_strs=filter_strings)
    print(len(result[1]) / (12 * 7))

    for well in result[0]:
        print('######################', well)
        for tup in result[0][well]:
            print(tup, result[0][well][tup])

    # Beispielaufruf 2
    # print_directory_tree(base_directory, filter_strs=filter_strings)