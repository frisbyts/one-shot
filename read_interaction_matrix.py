import pandas as pd 
import numpy as np 
from sklearn.cluster.bicluster import SpectralBiclustering
from Bio import SubsMat
from Bio import AlignIO
from Bio import Alphabet
from Bio.Alphabet import IUPAC
from Bio.Align import AlignInfo

class process_data:
    def __init__(self,interaction_matrix,n_clusters,kinase_msa,peptide_msa):
        self.interaction_matrix = interaction_matrix
        self.n_clusters = n_clusters
        self.kinase_msa = kinase_msa
        self.peptide_msa = peptide_msa

    def _spectral_bicluster(self,n_clusters,interaction_matrix):
        clustering = SpectralBiclustering(n_clusters=n_clusters,random_state=0).fit(self.interaction_matrix)
        pdz_clusters = clustering.row_labels_
        peptide_clusters = clustering.column_labels_
        return pdz_clusters, peptide_clusters 

    def _listify_msa(self,msa):
        alignments = []
        with open(msa,"r") as seqs:
            for line in seqs:
                line = line.strip("\n").split("\t")
                seq = line[1]
                alignments.append(seq)
        return alignments

    def substitution_matrices(self):
        subs = []
        for msa in [self.kinase_msa,self.peptide_msa]:
            c_align = AlignIO.read(msa, "tab")
            summary_align = AlignInfo.SummaryInfo(c_align)
            replace_info = summary_align.replacement_dictionary()
            my_arm = SubsMat.SeqMat(replace_info)
            my_lom = SubsMat.make_log_odds_matrix(my_arm)
            subs.append(my_lom)
        return subs[0], subs[1] 

    def _get_encoding(self,seq,sub_matrix):
        encoding_from_sub_matrix = []
        alpha = ["-","A","C","D","E","F","G",
                 "H","I","K","L","M","N","P",
                 "Q","R","S","T","V","W","Y"]
        for aa in seq:
            for sym in alpha:
                if (aa,sym) in sub_matrix:
                    encoding_from_sub_matrix.append(sub_matrix[(aa,sym)])
                else:
                    encoding_from_sub_matrix.append(sub_matrix[(sym,aa)])
        return encoding_from_sub_matrix

    def encode_data(self):
        pdz_clusters, peptide_clusters = self._spectral_bicluster(self.n_clusters,self.interaction_matrix)
        





if __name__ == '__main__':

    t = pd.read_csv("t.csv",header=None)
    p = process_data(t,5,"mouse_pdz.msa","mouse_peptide.msa")
    m, _ = p.substitution_matrices()
    print(m)

    




