# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 22:48:55 2020

@author: jaces
Source:
    https://python-graph-gallery.com/wordcloud/
    https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html#wordcloud.WordCloud
"""

###############################################################################
##  Libraries
###############################################################################
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image    #pillow library to import the image -->conda install -c anaconda pillow
from wordcloud import WordCloud


###############################################################################
## Making a basic wordcloud
###############################################################################
# Create a list of word
text=("Python Python Python Matplotlib Matplotlib Seaborn Network Plot Violin Chart Pandas Datascience Wordcloud Spider Radar Parrallel Alpha Color Brewer Density Scatter Barplot Barplot Boxplot Violinplot Treemap Stacked Area Chart Chart Visualization Dataviz Donut Pie Time-Series Wordcloud Wordcloud Sankey Bubble")
 
# Create the wordcloud object
wordcloud = WordCloud(width=480, height=480, margin=0).generate(text)
# To change font size use the next line
#wordcloud = WordCloud(width=480, height=480, max_font_size=20, min_font_size=10).generate(text)
# To define the number of words
#wordcloud = WordCloud(width=480, height=480, max_words=3).generate(text)
# To remove some words
#wordcloud = WordCloud(width=480, height=480, stopwords=["Python", "Matplotlib"]).generate(text)
# To change color background
#wordcloud = WordCloud(width=480, height=480, background_color="skyblue").generate(text)
# To change color of words
#wordcloud = WordCloud(width=480, height=480, colormap="Blues").generate(text)


# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()



###############################################################################
## Making a Worcloud with specific shape
###############################################################################
# Create a list of word (https://en.wikipedia.org/wiki/Data_visualization)
text=("Data visualization or data visualisation is viewed by many disciplines as a modern equivalent of visual communication. It involves the creation and study of the visual representation of data, meaning information that has been abstracted in some schematic form, including attributes or variables for the units of information A primary goal of data visualization is to communicate information clearly and efficiently via statistical graphics, plots and information graphics. Numerical data may be encoded using dots, lines, or bars, to visually communicate a quantitative message.[2] Effective visualization helps users analyze and reason about data and evidence. It makes complex data more accessible, understandable and usable. Users may have particular analytical tasks, such as making comparisons or understanding causality, and the design principle of the graphic (i.e., showing comparisons or showing causality) follows the task. Tables are generally used where users will look up a specific measurement, while charts of various types are used to show patterns or relationships in the data for one or more variables")
 
# Load the image (http://python-graph-gallery.com/wp-content/uploads/wave.jpg)
wave_mask = np.array(Image.open("wave.jpg"))
 
# Make the figure
wordcloud = WordCloud(mask=wave_mask, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()
plt.style.use('default')