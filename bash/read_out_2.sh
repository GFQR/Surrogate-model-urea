#!/bin/bash

q_dist="dip"
q_magn="ch01"
file="qr_urea_dip_ch01_rot"
output="beta_YYY_dip_ch01.txt"

# Clear output file at start
> "$output"

printf "$file\n" >> "$output"
printf "q_dist, q_magn, angle, beta(Y;Y,Y)\n" >> "$output"

for k in 0 45 90 135 180 225 270 315
do
    # Extract numeric value after '=' using grep + awk
    value=$(grep 'beta(Y;Y,Y)' "${file}${k}.out" | awk -F'Y) =' '{print $2}' | awk '{print $1}')

    # If found, print angle and value
    if [ -n "$value" ]; then
        printf "%s, %s, %s, %s\n" "$q_dist" "$q_magn" "$k" "$value" >> "$output"
    else
        printf "%s, N/A\n" "$k" >> "$output"
    fi
done

echo "done"
