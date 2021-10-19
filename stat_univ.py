# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 12:39:26 2021

@author: pierr
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Statistiques univariées des images
dir_images = "./projet_python"
categ = ["Arbre", "Fleur", "Autre"]

photos = []
images = []
for cat in categ:
    content = os.listdir(dir_images+"/"+cat)
    sumP = 0  # nb photos
    sumI = 0  # nb images
    for file in content:
        if file[0] == "P":
            sumP += 1
        else :
            sumI += 1
    photos.append(sumP)
    images.append(sumI)


df = pd.DataFrame({"Photos": photos,
                   "Images": images},
                   index=categ)
print(df)
df.plot.bar(rot=0, ylim=(0, max(df.Images)+10),
            color={"Photos": "forestgreen", "Images": "teal"})

plt.savefig("bar_plot.png", dpi=200)
