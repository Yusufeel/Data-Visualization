#!/usr/bin/env python
# coding: utf-8

# Sapma
# 
# Sabit bir referans noktasından varyasyonları (+/-) vurgulayın. Tipik olarak referans noktası sıfırdır, ancak aynı zamanda bir hedef veya uzun vadeli bir ortalama da olabilir. Duygu göstermek için de kullanılabilir (pozitif/nötr/negatif)

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

# Veri oluşturma
categories = ['A', 'B', 'C', 'D', 'E']
values = [-10, -5, 0, 5, 10]

# Renkleri belirleme
colors = ['red' if val < 0 else 'blue' for val in values]

# Bar grafiği oluşturma
plt.barh(categories, values, color=colors)

# Eksen etiketlerini ve başlığı ayarlama
plt.xlabel('Magnitude')
plt.title('Diverging Bar Chart')

# Sıfır çizgisini eklemek için
plt.axvline(0, color='black', linestyle='--', linewidth=1)

# Grafiği gösterme
plt.show()


# Ayrışan yığılmış çubuk
# 
# Duyarlılık içeren anket sonuçlarını sunmak için mükemmel (örneğin katılmıyorum, tarafsız, kabul edildi

# In[2]:


import matplotlib.pyplot as plt
import numpy as np

# Veri oluşturma
categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
positive_values = np.array([20, 30, 15, 25])
negative_values = np.array([-10, -15, -5, -20])

# Renkleri belirleme
colors = ['green', 'lightgreen', 'lightgrey', 'lightcoral']

# Diverging stacked bar grafiği oluşturma
fig, ax = plt.subplots()
ax.bar(categories, positive_values, color=colors)
ax.bar(categories, negative_values, color=colors)

# Eksen etiketlerini ve başlığı ayarlama
plt.xlabel('Categories')
plt.ylabel('Magnitude')
plt.title('Diverging Stacked Bar Chart')

# Sıfır çizgisini eklemek için
plt.axhline(0, color='black', linestyle='--', linewidth=1)

# Grafiği gösterme
plt.show()

Spine grafiği, tek bir değeri iki zıt bileşene (örneğin, Erkek/Kadın gibi) ayırmak için kullanılan bir görselleştirme yöntemidir. Bu grafiği oluşturmak için aşağıdaki adımları takip edebilirsiniz:
# In[7]:


import matplotlib.pyplot as plt
import numpy as np

# Örnek veri oluşturma
ulkeler = ['Ülke A', 'Ülke B', 'Ülke C', 'Ülke D']
erkekler = np.array([45, 30, 55, 40])
kadinlar = 100 - erkekler  # Toplam nüfusun yüzde 100 olduğunu varsayalım

# Spine grafiği oluşturma
fig, ax = plt.subplots()

# Erkekleri temsil eden sol bileşen
ax.barh(ulkeler, erkekler, color='blue', label='Erkek')

# Kadınları temsil eden sağ bileşen
ax.barh(ulkeler, kadinlar, left=erkekler, color='pink', label='Kadın')

# Eksen etiketleri ve başlık
plt.xlabel('Nüfus Yüzdesi')
plt.title('Ülkelere Göre Erkek ve Kadın Dağılımı')

# Gösterge ekleme
plt.legend()

# Grafiği gösterme
plt.show()



# Surplus/deficit filled line grafiği, iki zaman serisi arasındaki fazlalık (surplus) veya eksiklik (deficit) miktarını görsel olarak ifade etmek için kullanılan bir grafik türüdür. Bu grafiğin önemli bir özelliği, iki serinin arasındaki dengenin veya baz çizgisine göre olan fazlalığın/eksikliğin gölgeli bir alan ile vurgulanmasıdır.

# In[2]:


import matplotlib.pyplot as plt
import numpy as np

# Veri oluşturma
x_values = np.array([1, 2, 3, 4, 5])
series1 = np.array([10, 8, 12, 6, 14])
series2 = np.array([8, 10, 6, 12, 4])

# Surplus/deficit hesaplama
surplus_deficit = series1 - series2

# Surplus ve deficit değerlerini kullanarak dolgu grafiği oluşturma
plt.fill_between(x_values, surplus_deficit, color='lightgray', alpha=0.5, label='Surplus/Deficit')

# İlk seri çizgi grafiği
plt.plot(x_values, series1, label='Series 1', marker='o')

# İkinci seri çizgi grafiği
plt.plot(x_values, series2, label='Series 2', marker='o')

# Başlık ve etiketler
plt.title('Surplus/Deficit Filled Line Chart')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# Göstergeyi ekleme
plt.legend()

# Grafiği gösterme
plt.show()


# # Correlation

# İki veya daha fazla değişken arasındaki ilişkiyi göster. Onlara aksini söylemediğiniz sürece, birçok okuyucunun onlara gösterdiğiniz ilişkilerin nedensel olduğunu varsayacağını unutmayın (yani biri diğerine neden olur)

# In[8]:


import matplotlib.pyplot as plt
import numpy as np

# Zamanı temsil eden veri
zaman = np.arange(1, 6)

# Rastgele örneklenmiş miktar verisi
miktar = np.random.randint(80, 120, size=5)

# Rastgele örneklenmiş oran verisi
oran = np.random.uniform(0.5, 1.0, size=5)

# Sütun grafiği oluşturma
plt.bar(zaman, miktar, color='blue', label='Miktar')

# Çizgi grafiği oluşturma
plt.plot(zaman, oran, color='red', marker='o', label='Oran')

# Eksen etiketleri ve başlık
plt.xlabel('Zaman')
plt.ylabel('Miktar ve Oran')
plt.title('Zaman İçinde Miktar ve Oran İlişkisi')

# Sütun ve çizgi grafiğini bir araya getirmek için ikinci y eksen
ax2 = plt.twinx()
ax2.set_ylabel('Oran', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Gösterge ekleme
plt.legend()

# Grafiği gösterme
plt.show()




# Bağlantılı scatter plot, genellikle iki değişken arasındaki ilişkinin zaman içinde nasıl değiştiğini göstermek için kullanılan bir grafik türüdür. Bu tür bir grafiği oluşturmak için kullanılan Python kodunu aşağıda bulabilirsiniz:

# In[5]:


import matplotlib.pyplot as plt
import numpy as np

# Zamanı temsil eden örnek veri
zaman = np.array([1, 2, 3, 4, 5])

# İki değişkeni temsil eden örnek veri
degisken1 = np.array([10, 20, 15, 25, 30])
degisken2 = np.array([5, 15, 10, 20, 25])

# Bağlantılı scatter plot oluşturma
plt.plot(zaman, degisken1, marker='o', linestyle='-', color='blue', label='Değişken 1')
plt.plot(zaman, degisken2, marker='o', linestyle='-', color='green', label='Değişken 2')

# Eksen etiketleri ve başlık
plt.xlabel('Zaman')
plt.ylabel('Değişken Değerleri')
plt.title('Bağlantılı Scatter Plot - İki Değişkenin Zaman İçindeki İlişkisi')

# Gösterge ekleme
plt.legend()

# Grafiği gösterme
plt.show()


# Bubble grafiği, bir scatter plot gibi davranır ancak daha fazla ayrıntı ekler, çünkü dairelerin boyutları üçüncü bir değişkene bağlı olarak belirlenir. Bu grafiğin amacı, iki değişken arasındaki ilişkiyi gösterirken, aynı zamanda üçüncü bir değişkenin değerini dairelerin büyüklüğüyle görsel olarak ifade etmektir.
# 
# Başka bir deyişle, her bir dairenin konumu (x, y koordinatları) iki değişkeni temsil ederken, dairenin boyutu üçüncü bir değişkeni temsil eder. Bu, bir bakışta üç farklı değişken arasındaki ilişkiyi anlamamıza yardımcı olur.

# In[10]:


import matplotlib.pyplot as plt
import numpy as np

# Örnek veri oluşturma
x = np.random.rand(20)
y = np.random.rand(20)
size = np.random.rand(20) * 100  # Üçüncü değişkeni temsil etmek için boyutları belirle

# Bubble grafiği oluşturma
plt.scatter(x, y, s=size, alpha=0.7, c='blue', edgecolors='black')

# Eksen etiketleri ve başlık
plt.xlabel('X Değişkeni')
plt.ylabel('Y Değişkeni')
plt.title('Bubble Grafiği - Üçüncü Değişkenin Boyutlarıyla')

# Grafiği gösterme
plt.show()


# In[11]:


import matplotlib.pyplot as plt
import numpy as np

# Rastgele örnek veri oluşturma
yas_araliklari = ['18-25', '26-35', '36-45', '46-55', '56+']
kategori1_satis = np.random.randint(50, 150, size=len(yas_araliklari))
kategori2_satis = np.random.randint(40, 130, size=len(yas_araliklari))

# Grafik oluşturma
fig, ax = plt.subplots()

# Çizgi grafiği ile deseni gösterme
ax.plot(yas_araliklari, kategori1_satis, label='Kategori 1', marker='o')
ax.plot(yas_araliklari, kategori2_satis, label='Kategori 2', marker='o')

# Eksen etiketleri ve başlık
plt.xlabel('Yaş Aralığı')
plt.ylabel('Satış Miktarı')
plt.title('Yaş Aralıklarına Göre Satış Miktarı')

# Gösterge ekleme
plt.legend()

# Grafiği gösterme
plt.show()


# In[14]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
ages = np.random.randint(18, 60, size=(1000,))
savings = np.random.uniform(1000, 50000, size=(1000,))

df = pd.DataFrame({'Age': ages, 'Saving': savings})

age_bins = [18, 25, 35, 45, 60]
age_labels = ['18-25', '26-35', '36-45', '46+']

# Yaş aralıklarına göre verileri gruplayın ve ortalamalarını alın
df['Yaş Aralığı'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
grouped_data = df.groupby('Yaş Aralığı')['Saving'].mean().reset_index()

# Veriyi heatmap için düzenleyin
heatmap_data = pd.pivot_table(df, values='Saving', index='Yaş Aralığı', aggfunc='mean').reset_index()


sns.heatmap(heatmap_data.set_index('Yaş Aralığı'), annot=True, fmt=".2f", cmap="YlGnBu")

# Grafiği gösterin
plt.title('Average Saving Amount by Age Range')
plt.xlabel('Yaş Aralığı')
plt.ylabel('Average Saving Amount')
plt.show()


# In[ ]:




