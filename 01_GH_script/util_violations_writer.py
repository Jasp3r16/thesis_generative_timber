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
underused  = sorted([float(v) for v in util_values if float(v) < 1.0])
count      = len(violations)
count_u    = len(underused)

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
            max_v  = max(1, count)
            max_u  = max(1, count_u)
            header = (['rank', 'count_above_1'] +
                      ['util_above_{}'.format(i + 1) for i in range(max_v)] +
                      ['count_below_1'] +
                      ['util_below_{}'.format(i + 1) for i in range(max_u)])
            writer.writerow(header)

        row = ([rank, count] +
               [round(v, 6) for v in violations] +
               [count_u] +
               [round(v, 6) for v in underused])
        writer.writerow(row)

    print("Written rank {} to {} — {} above 1, {} below 1.".format(rank, output_path, count, count_u))
else:
    print("Rank {} — {} above 1, {} below 1 (not written).".format(rank, count, count_u))

# ==========================================
# OUTPUTS
# ==========================================
Violations = violations
Count      = count
Underused  = underused
CountBelow = count_u
