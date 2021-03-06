# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 22:51:24 2021

@author: jaces
"""

from legibilidad import legibilidad

##############################################################################
## pkg: C:\Anaconda3\envs\datascience\Lib\site-packages\legibilidad\legibilidad.py
##############################################################################

texto = 'Legibilidad, se refiere a la facilidad para leer un texto'
print(legibilidad.inflesz(legibilidad.szigriszt_pazos(texto)))

TextoDePrueba = '''
Tuvo muchas veces competencia con el cura de su lugar (que era hombre docto graduado en Sigüenza), sobre cuál había sido mejor caballero, Palmerín de Inglaterra o Amadís de Gaula; mas maese Nicolás, barbero del mismo pueblo, decía que ninguno llegaba al caballero del Febo, y que si alguno se le podía comparar, era don Galaor, hermano de Amadís de Gaula, porque tenía muy acomodada condición para todo; que no era caballero melindroso, ni tan llorón como su hermano, y que en lo de la valentía no le iba en zaga.
En resolución, él se enfrascó tanto en su lectura, que se le pasaban las noches leyendo de claro en claro, y los días de turbio en turbio, y así, del poco dormir y del mucho leer, se le secó el cerebro, de manera que vino a perder el juicio. Llenósele la fantasía de todo aquello que leía en los libros, así de encantamientos, como de pendencias, batallas, desafíos, heridas, requiebros, amores, tormentas y disparates imposibles, y asentósele de tal modo en la imaginación que era verdad toda aquella máquina de aquellas soñadas invenciones que leía, que para él no había otra historia más cierta en el mundo.
'''
print(legibilidad.read_speed(TextoDePrueba))
