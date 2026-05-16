import csv
import os

# ==========================================
# INPUTS
# ==========================================
util_values = globals().get("util_values")   # flat list of floats
rank        = int(globals().get("rank", 0))
file_path   = globals().get("file_path")
write       = bool(globals().get("write", False))

# ==========================================
# COMPUTE
# ==========================================
violations = sorted([float(v) for v in util_values if float(v) > 1.0], reverse=True)
count      = len(violations)

# ==========================================
# WRITE
# ==========================================
if write and file_path:
    ga_prefix   = os.path.basename(file_path).split('_top10')[0]
    output_path = os.path.join(os.path.dirname(file_path), ga_prefix + '_util_violations.csv')
    file_exists = os.path.isfile(output_path)

    with open(output_path, 'a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            max_cols = max(1, count)
            header   = ['rank', 'count_above_1'] + ['util_{}'.format(i + 1) for i in range(max_cols)]
            writer.writerow(header)

        row = [rank, count] + [round(v, 6) for v in violations]
        writer.writerow(row)

    print("Written rank {} to {} — {} violation(s).".format(rank, output_path, count))
else:
    print("Rank {} — {} violation(s) above 1 (not written).".format(rank, count))

# ==========================================
# OUTPUTS
# ==========================================
Violations = violations
Count      = count
