import re
import csv
from collections import Counter
path = r'C:\\ProcessData\\src_code\\Testing\\analyze_loss\\val_loss_rank.csv'
pattern = re.compile(r'C\d{4}')
counter = Counter()
with open(path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        match = pattern.search(row['image'])
        if match:
            counter[match.group()] += 1
for key in sorted(counter):
    print(f"{key},{counter[key]}")
print("TOTAL", sum(counter.values()))
