#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('nyc_taxi (1).csv', usecols=\
                ['pickup_x','pickup_y','dropoff_x','dropoff_y','passenger_count','tpep_pickup_datetime'])
df.tail() #son beş veriyi alır.


# Web Mercator kordinatlarından alınan verilerdir.

# In[2]:


# NumPy kütüphanesini, sayısal işlemler ve diziler için dahil ediyoruz.
import numpy as np

# HoloViews kütüphanesini, görselleştirme için dahil ediyoruz.
import holoviews as hv

# HoloViews'teki görselleştirme seçeneklerini kullanmak için dahil ediyoruz.
from holoviews import opts

# EsriStreet türünü kullanarak harita görselleştirmesi için dahil ediyoruz.
from holoviews.element.tiles import EsriStreet

# HoloViews'ün Bokeh çıktısını etkinleştiriyoruz.
hv.extension('bokeh')


# In[3]:


from bokeh.models import BoxZoomTool  # Kutu Zoom aracını içe aktar
from bokeh.plotting import figure, output_notebook, show  # Figür oluşturma ve gösterim için gerekli araçları içe aktar

output_notebook()  # Bokeh çizimini Jupyter Notebook içinde görüntülemek için ayarla

# New York City'nin koordinat aralığını tanımla
NYC = x_range, y_range = ((-8242000, -8210000), (4965000, 4990000))

# Çizimin genişliğini ve yüksekliğini belirle
plot_width = int(990)
plot_height = int(plot_width // 1.2)  # Yüksekliği genişliğe orantılı olarak ayarla

# Çizim öğelerinin (noktalar, çizgiler vb.) varsayılan stil ayarlarını belirle
options = dict(line_color=None, fill_color='blue', size=5)


# In[4]:


# Tekrar kullanılabilir, basit Bokeh çizimleri oluşturmak için bir fonksiyon tanımla
def base_plot(tools='pan,wheel_zoom,reset', plot_width=plot_width, plot_height=plot_height, **plot_kwargs):
    # Temel bir Bokeh çizimi oluştur
    p = figure(
        tools=tools,  # Kullanılacak etkileşim araçlarını ayarla (kaydırma, tekerlek ile yakınlaştırma, sıfırlama)
        plot_width=plot_width,  # Çizimin genişliğini ayarla
        plot_height=plot_height,  # Çizimin yüksekliğini ayarla
        x_range=x_range,  # Çizimin x ekseninin kapsamını belirle
        y_range=y_range,  # Çizimin y ekseninin kapsamını belirle
        outline_line_color=None,  # Çizimin dış çerçevesini gizle
        min_border=0, min_border_left=0, min_border_right=0, min_border_top=0, min_border_bottom=0,  # Çizimin kenar boşluklarını kaldır
        **plot_kwargs  # Diğer opsiyonel çizim ayarlarını kabul et
    )

    # Eksenleri ve ızgara çizgilerini gizle
    p.axis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    # Kutu Zoom aracını ekle (en-boy oranını koruyarak yakınlaştırma)
    p.add_tools(BoxZoomTool(match_aspect=True))

    # Oluşturulan çizimi döndür
    return p



# In[5]:


# Bokeh kütüphanesinden, çizim ve gösterim fonksiyonlarını dahil ediyoruz.
from bokeh.plotting import figure, show

# Bokeh'in sunduğu harita altlık sağlayıcılarını kullanmak için dahil ediyoruz.
from bokeh.tile_providers import get_provider, STAMEN_TERRAIN

# Veri setinden 1000 örneklem alıyoruz.
samples = df.sample(n=1000)

# Temel çizimi oluşturuyoruz.
p = base_plot()

# STAMEN_TERRAIN harita altlığını ayarlıyoruz.
tile_provider = get_provider(STAMEN_TERRAIN)

# Çizime harita altlığını ekliyoruz.
p.add_tile(tile_provider)

# Veri noktalarını çemberler olarak çiziyoruz.
p.circle(x=samples['dropoff_x'], y=samples['dropoff_y'], **options)

# Çizimi gösteriyoruz.
show(p)


# Manhattının merkezindeki gerçek taksiden indrilme yoğunlunu oklüzyon nedeniyle görmek imkansız

# In[6]:


# Veri setinden 10000 örneklem alıyoruz.
samples = df.sample(n=10000)

# Temel çizimi oluşturuyoruz.
p = base_plot()

# STAMEN_TERRAIN harita altlığını ayarlıyoruz.
tile_provider = get_provider(STAMEN_TERRAIN)

# Çizime harita altlığını ekliyoruz.
p.add_tile(tile_provider)

# Veri noktalarını çemberler olarak çiziyoruz.
p.circle(x=samples['dropoff_x'], y=samples['dropoff_y'], **options)

# Çizimi gösteriyoruz.
show(p)


# In[7]:


import datashader as ds

#Datashaderın transfer fonksiyonları(renklendirme, ölçekleme) kullanacağımız kütüphaneleri import edeşm,
from datashader import transfer_functions as tf
#Renk paletlerini import edelim
from datashader.colors import Greys9

#Renk paletlerini ters çevirerek yeni bir palet oluşturuyoruz
Greys9_r = list(reversed(Greys9))[:-2] # son iki renki çıkardık


# In[8]:


#Datashaderın canvas'ı oluşturuyoruz.
cvs = ds.Canvas(plot_width=plot_width, plot_height=plot_height, x_range=x_range, y_range=y_range)

#Veri Noktalarını canvas üzerinde çizelim
agg = cvs.points(df, 'dropoff_x','dropoff_y')

img = tf.shade(agg, cmap=["white",'darkblue'], how="linear")
img


# In[9]:


frequencies,edges = np.histogram(agg.values, bins=100)
hv.Histogram((edges, frequencies)).opts(width=800).redim.range(Frequency=(0,6000))


# In[10]:


frequencies,edges = np.histogram(np.log1p(agg.values), bins=100)
hv.Histogram((edges, frequencies)).opts(width=800).redim.range(Frequency=(0,8000))


# In[13]:


img = tf.shade(agg, cmap=Greys9_r, how='log')
img


# In[15]:


frequencies,edges = np.histogram(tf.eq_hist(agg.values), bins=100)
hv.Histogram((edges, frequencies)).opts(width=800).redim.range(Frequency=(0,6000))


# In[16]:


img = tf.shade(agg, cmap=Greys9_r, how='eq_hist')
img


# In[21]:


# HoloViews'in Datashader ile olan entegrasyonunu dahil ediyoruz:
import holoviews.operation.datashader as hd

# Colorcet renk paletlerini kullanmak için dahil ediyoruz:
import colorcet as cc

# Datashader kullanarak büyük veri setini görselleştiriyoruz:

# 1. Veri noktalarını HoloViews Points nesnesi olarak tanımlıyoruz:
points = hv.Points(df, ['dropoff_x', 'dropoff_y'])

# 2. Datashader'ın rasterize fonksiyonu ile görselleştiriyoruz:
shaded = hd.rasterize(points, aggregator=ds.sum('passenger_count'))  # Yolcu sayılarını toplayarak yoğunlukları hesaplıyoruz

# 3. Görselleştirmenin görünümünü ayarlıyoruz:
shaded.opts(
   cmap=cc.fire[100:],  # Colorcet'in "fire" paletinin bir kısmını renklendirme için kullanıyoruz
   cnorm='eq_hist',  # Veri değerlerini eşit oranda bölerek renklendiriyoruz
   nodata=0  # Veri olmayan alanları 0 değeriyle temsil ediyoruz
)

# 4. Datashader'ın dynspread fonksiyonu ile görselleştirmeyi optimize ediyoruz:
hd.dynspread(shaded, threshold=0.5, max_px=10).opts(
   bgcolor='black',  # Arka planı siyah yapıyoruz
   xaxis=None, yaxis=None,  # Eksenleri kaldırıyoruz
   width=900, height=500  # Çizimin boyutunu ayarlıyoruz
)


# In[26]:


#Veri görselleştirmesini özelleştirmek için bir fonksiyon tanımlayalım
def transform(img):
    #Görselleştirmek için en yoğun noktaları vurgulamak için bir transform edicez.
    agg = img.data["dropoff_x_dropoff_y Count"] # Yoğunluk değerlerini aldık
    return img.clone(agg.where(agg > np.percentile(agg, 90))) # yüzde 90'lık dilimden yüksek değerleri aldık

custom_points = hv.Points(df, ["dropoff_x", "dropoff_y"])

custom_rasterized = hd.rasterize(custom_points)

custom_shaded = hd.shade(custom_rasterized.apply(transform), cmap=cc.fire)

tiles = EsriStreet().redim.range(x=x_range, y=y_range).opts(alpha=0.5)
#Harita altlığı ile birleştirelim
tiles * hd.dynspread(custom_shaded, threshold=0.3, max_px=4) # harita altlığı üzerine görselleştirmeyi ekledik


# In[28]:


img = hd.rasterize(custom_points, dynamic=False)

img.data


# In[30]:


def transform(overlay):
    picks = overlay.get(0).redim(pickup_x='x',pickup_y='y')
    drops = overlay.get(1).redim(dropoff_x = 'x', dropoff_y='y')
    pick_agg = picks.data["pickup_x_pickup_y Count"].data
    drop_agg = drops.data["dropoff_x_dropoff_y Count"].data    
    more_picks = picks.clone(picks.data.where(pick_agg > drop_agg))
    more_drops = drops.clone(drops.data.where(drop_agg > pick_agg))
    return (hd.shade(more_drops, cmap=["lightcyan","blue"]) * 
            hd.shade(more_picks, cmap=["mistyrose","red"]))

picks = hv.Points(df, ["pickup_x","pickup_y"])
drops = hv.Points(df, ["dropoff_x","dropoff_y"])

((hd.rasterize(picks) * hd.rasterize(drops))).apply(transform).opts(
    bgcolor="white", xaxis=None, yaxis=None, width=900, height=500)


# In[ ]:




