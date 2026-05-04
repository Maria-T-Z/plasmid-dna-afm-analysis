# -*- coding: utf-8 -*-
from src import load_spm, flatten_and_correct, save_preprocessed, load_yolo_model, get_bounding_boxes, process_and_save_crops
import os
import glob


def process_all_files(input_dir: str, output_dir: str, model_path: str):
    """
    Główna pętla (Pipeline) przetwarzająca skany AFM.
    Od surowego pliku po maski Frangiego na wycinkach.
    """
    # 1. Znajdź pliki wejściowe
    search_pattern = os.path.join(input_dir, "*.spm")
    spm_files = glob.glob(search_pattern)

    if not spm_files:
        print(f"[UWAGA] Brak plików .spm w folderze: {input_dir}")
        return

    print(f"[START] Znaleziono {len(spm_files)} plików. Uruchamianie maszyny...")

    # 2. Wczytanie modelu YOLO (TYLKO RAZ - gigantyczna oszczędność czasu)
    try:
        model = load_yolo_model(model_path)
    except Exception as e:
        print(f"[BŁĄD FATALNY] Nie udało się wczytać modelu YOLO: {e}")
        return

    # Folder na wycięte plazmidy
    crops_dir = os.path.join(output_dir, "crops")

    # 3. Główna pętla przerabiająca każdy plik po kolei
    for i, file_path in enumerate(spm_files, 1):
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        print(f"\n[{i}/{len(spm_files)}] Przetwarzanie pliku: {filename}")

        try:
            # === FAZA 1: FIZYKA I PROSTOWANIE (io_utils) ===
            print("   -> Czytanie i prostowanie miki...")
            raw_data, scale, spm_file = load_spm(file_path)
            corrected_data = flatten_and_correct(raw_data)

            # Zapis pełnych obrazów
            save_preprocessed(corrected_data, scale, file_path, output_dir)

            # === FAZA 2: SZTUCZNA INTELIGENCJA (yolo_utils) ===
            # Odtwarzamy ścieżki do wygenerowanych przed chwilą plików
            npy_path = os.path.join(output_dir, "heights", f"{base_name}.npy")
            png_path = os.path.join(output_dir, "images", f"{base_name}.png")

            print("   -> Detekcja YOLO na pełnym obrazie...")
            boxes = get_bounding_boxes(model, png_path, conf_threshold=0.5)
            print(f"   -> Znaleziono {len(boxes)} potencjalnych plazmidów.")

            # === FAZA 3: CHIRURGICZNE CIĘCIE I FRANGI (crop_utils) ===
            if boxes:
                print("   -> Wycinanie kadrów i generowanie masek Frangiego...")
                saved_count = process_and_save_crops(npy_path, png_path, boxes, crops_dir, base_name)
                print(f"   -> Zapisano {saved_count} pełnych paczek kadrów (NPY, PNG, Maski).")
            else:
                print("   -> [INFO] YOLO nie znalazło nic ciekawego na tym skanie. Pomijam cięcie.")

        except Exception as e:
            print(f"   -> [BŁĄD] Plik {filename} wysypał się w trakcie procesu: {e}")

    print("\n[KONIEC] Taśma produkcyjna zakończyła pracę!")


# =====================================================================
# URUCHOMIENIE
# =====================================================================
if __name__ == "__main__":
    # Główne ścieżki (zgodnie z architekturą)
    FOLDER_WEJSCIOWY = "./data/raw_spm"
    FOLDER_WYJSCIOWY = "./data/processed"
    SCIEZKA_MODELU = "./models/best.pt"

    # Bezpieczniki: tworzymy foldery wejściowe, jeśli nie istnieją
    os.makedirs(FOLDER_WEJSCIOWY, exist_ok=True)
    os.makedirs("./models", exist_ok=True)

    # Odpalenie procesu!
    process_all_files(FOLDER_WEJSCIOWY, FOLDER_WYJSCIOWY, SCIEZKA_MODELU)