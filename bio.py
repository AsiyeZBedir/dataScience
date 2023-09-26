import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from Bio.SeqUtils import gc_fraction

# DNA dizilerini içeren bir FASTA dosyasını okuma
fasta_file = "dna_sequences.fasta"
sequences = [record.seq for record in SeqIO.parse(fasta_file, "fasta")]

# Dizilerin uzunluklarını hesaplama
sequence_lengths = [len(seq) for seq in sequences]

# Dizilerin GC içeriğini hesaplama
gc_contents = [gc_fraction(seq) for seq in sequences]

# DNA dizilerinin istatistiksel analizi
mean_length = np.mean(sequence_lengths)
mean_gc_content = np.mean(gc_contents)

# Sonuçları görselleştirme
plt.figure(figsize=(12, 6))

# Dizilerin uzunluklarının histogramı
plt.subplot(1, 2, 1)
plt.hist(sequence_lengths, bins=20, color='blue', alpha=0.7)
plt.xlabel('Dizi Uzunluğu')
plt.ylabel('Frekans')
plt.title('Dizi Uzunluğu Dağılımı')

# Dizilerin GC içeriğinin kutu grafiği
plt.subplot(1, 2, 2)
plt.boxplot(gc_contents)
plt.ylabel('GC İçeriği')
plt.title('GC İçeriği Kutu Grafiği')

plt.tight_layout()
plt.show()

# Veri analizi sonuçlarını raporlama
print("Ortalama Dizi Uzunluğu:", mean_length)
print("Ortalama GC İçeriği:", mean_gc_content)

