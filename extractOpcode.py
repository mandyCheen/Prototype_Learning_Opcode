import pandas as pd
import r2pipe as r2
import concurrent.futures
from multiprocessing import Value, Manager
import os
import hashlib
import time

DATASET_PATH = "/home/mandy900619/data/Malware202403_info.csv"
DATASET_FOLDER = "/home/mandy900619/data/Malware202403/"
ERROR_PATH = "./log/error_5_9_4_text.log"
DUPLICATE_PATH = "./log/duplicate_5_9_4_text.log"
MAX_FILES_PER_FAMILY = 20
MAX_WORKERS = 15
SEED = 7
TIME_LIMIT = 1800

print(f"Loading dataset from {DATASET_PATH}...")
dataset = pd.read_csv(DATASET_PATH)
dataset_unpacked = dataset[dataset['is_packed'] == False]
dataset_unpacked = dataset_unpacked.dropna(subset=['family'])
dataset_unpacked = dataset_unpacked.dropna(subset=['CPU'])
dataset_unpacked = dataset_unpacked[dataset_unpacked['CPU'] != "<unknown>"]
dataset_unpacked = dataset_unpacked[~dataset_unpacked['family'].str.contains("SINGLETON:")]
dataset_family_counts = dataset_unpacked['family'].value_counts()
# Filter out families with less than 20 samples
dataset_unpacked = dataset_unpacked[dataset_unpacked['family'].isin(dataset_family_counts[dataset_family_counts >= 20].index)]
dataset_unpacked = dataset_unpacked.sample(frac=1, random_state=SEED).reset_index(drop=True)

total = len(dataset_unpacked)
print(f"Total files to process: {total}")

def get_opcode_hash(opcode):
    return hashlib.md5(opcode.encode()).hexdigest()

def process_row(row, files_counter, family_counters, family_hashes):
    cpu = row.CPU
    family = row.family
    file_name = row.file_name
    output_folder = f"./data_5_9_4_text/{cpu}/{family}/"
    output_path = os.path.join(output_folder, f"{file_name}.txt")

    key = f"{cpu}/{family}"

    # Check if we've already processed 50 files for this CPU/family combination
    if family_counters[f"{cpu}/{family}"] >= MAX_FILES_PER_FAMILY:
        with files_counter.get_lock():
            files_counter.value += 1
            if files_counter.value % 10 == 0:
                print(f"Remaining files: {files_counter.value} / {total}", end="\r")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_path = os.path.join(DATASET_FOLDER, row.file_name[:2], row.file_name)

    try:
        opcodeAnalysis = r2.open(file_path, flags=["-2"])
        opcodeAnalysis.cmd("aaa")
        sections = opcodeAnalysis.cmdj('iSj')
        opcodes = ""
        if sections:
            for section in sections:
                if section['name'] == ".text": # Check if section is .text
                    if section['size'] > 0:
                        opcodes += opcodeAnalysis.cmd(f"pI {section['size']} @ {section['vaddr']} ~[0] ~!invalid")
            if opcodes == "":
                open(ERROR_PATH, "a").write(f"{row.file_name} has no opcodes\n")
            else:
                opcode_hash = get_opcode_hash(opcodes)

                if opcode_hash not in family_hashes[key]:
                    with open(output_path, "w") as f:
                        f.write(opcodes)
                    family_hashes[key][opcode_hash] = True
                    family_counters[key] += 1
                else:
                    open(DUPLICATE_PATH, "a").write(f"{row.file_name} is a duplicate \n")

        else:
            open(ERROR_PATH, "a").write(f"{row.file_name} has no sections\n")


        opcodeAnalysis.quit()

    except Exception as e:
        with open(ERROR_PATH, "a") as error_file:
            error_file.write(f"{row.file_name} failed: {str(e)}\n")

    with files_counter.get_lock():
        files_counter.value += 1
        if files_counter.value % 10 == 0:
            print(f"Remaining files: {files_counter.value} / {total}", end="\r")
    return 

if __name__ == "__main__":

    files_counter = Value('i', 0)
    
    with Manager() as manager:
        family_counters = manager.dict()
        family_hashes = manager.dict()
        # Initialize counters and hash sets for each CPU/family combination
        unique_cpu_family = dataset_unpacked.groupby(['CPU', 'family']).size().reset_index()[['CPU', 'family']]
        
        for _, row in unique_cpu_family.iterrows():
            key = f"{row.CPU}/{row.family}"
            if key not in family_counters:
                family_counters[key] = 0
                family_hashes[key] = manager.dict()
            output_folder = f"./data_5_9_4_text/{key}/"
            if os.path.exists(output_folder):
                files = []
                for file in os.listdir(output_folder):
                    with open(os.path.join(output_folder, file), "r") as f:
                        opcode_hash = get_opcode_hash(f.read())
                        family_hashes[key][opcode_hash] = True
                        family_counters[key] += 1
                        files.append(file.split(".")[0])
                dataset_unpacked = dataset_unpacked[~((dataset_unpacked['file_name'].isin(files)))]
                if family_counters[key] >= MAX_FILES_PER_FAMILY:
                    dataset_unpacked = dataset_unpacked[~((dataset_unpacked['CPU'] == row.CPU) & (dataset_unpacked['family'] == row.family))]
            
        print("Existing files loaded.")
        if os.path.exists(ERROR_PATH):
            with open(ERROR_PATH, "r") as error_file:
                file_errors = error_file.readlines()
                error_files = [file.split(" ")[0] for file in file_errors]
                print(f"Error files loaded: {len(error_files)}")
                dataset_unpacked = dataset_unpacked[~dataset_unpacked['file_name'].isin(error_files)]
        if os.path.exists(DUPLICATE_PATH):
            with open(DUPLICATE_PATH, "r") as duplicate_file:
                file_duplicates = duplicate_file.readlines()
                duplicate_files = [file.split(" ")[0] for file in file_duplicates]
                print(f"Duplicate files loaded: {len(duplicate_files)}")
                dataset_unpacked = dataset_unpacked[~dataset_unpacked['file_name'].isin(duplicate_files)]
        print("Processing files...")
        total = len(dataset_unpacked)
        print(f"Total files to process: {total}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_row, row, files_counter, family_counters, family_hashes): row 
                       for row in dataset_unpacked.itertuples()}
            
            for future in concurrent.futures.as_completed(futures):
                row = futures[future]
                try:
                    future.result(timeout=TIME_LIMIT)  # 設置超時時間
                except concurrent.futures.TimeoutError:
                    print(f"Processing of file {row.file_name} timed out after {TIME_LIMIT} seconds")
                    with open(ERROR_PATH, "a") as error_file:
                        error_file.write(f"{row.file_name} timed out after {TIME_LIMIT} seconds\n")
                except Exception as exc:
                    print(f"Processing of file {row.file_name} generated an exception: {exc}")
                    with open(ERROR_PATH, "a") as error_file:
                        error_file.write(f"{row.file_name} failed: {str(exc)}\n")

    print("\nProcessing completed.")