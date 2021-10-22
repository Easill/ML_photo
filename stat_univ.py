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
dir_traintest = ["Test_Data", "Training_Data"]
categ = ["Arbre", "Fleur", "Autre"]

# photos = []
# images = []
# for dir_data in dir_traintest:
#     sumP = 0
#     sumI = 0
#     for cat in categ:
#         content = os.listdir(dir_images+"/"+dir_data+"/"+cat)
#         sumP_dir = 0  # nb photos
#         sumI_dir = 0  # nb images
#         for file in content:
#             if file[0] == "P":
#                 sumP_dir += 1
#             else :
#                 sumI_dir += 1
        
    # photos.append(sumP)
    # images.append(sumI)

# print("Nb. photos :", sum(photos))
# print("Nb. images :", sum(images))

photos = [32, 16, 20]
images = [1542, 1503, 1577]


df = pd.DataFrame({"Photos": photos,
                   "Images": images},
                   index=categ)
print(df)
df.plot.bar(rot=0, ylim=(0, max(df.Images)+10),
            color={"Photos": "forestgreen", "Images": "purple"})

plt.savefig("bar_plot.png", dpi=200)
