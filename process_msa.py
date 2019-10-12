import numpy as np 
import matplotlib.pyplot as plt 
import sys

subfamily_counts = {}
pdz_msa = "PF00595_full.txt"
pdz_mouse_msa = "PF00595_selected_sequences.fa"
with open(pdz_mouse_msa,"r") as pdz:
	for line in pdz:
		if line.startswith(">"):
			try:
			    subfamily = line.strip("\n").split("{")[1].split("|")[1]
			except:
			    subfamily = line.strip("\n").split("/")[0]
			#subfamily = line.strip("\n").split("_")[1].split("/")[0]
			if subfamily_counts.get(subfamily):
				subfamily_counts[subfamily] += 1
			else:
				subfamily_counts[subfamily] = 1


print(len(subfamily_counts))
counts = list(subfamily_counts.values())
plt.hist(counts,bins=50)
plt.xlabel("Number of sequences")
plt.ylabel("Number of subfamilies")
plt.title("All species PDZ sequences")
#plt.savefig("all_species_pdz_sequences.png")
plt.show()

count = 0
for fam in subfamily_counts:
	if subfamily_counts[fam] > 1:
		count += 1

print(count)