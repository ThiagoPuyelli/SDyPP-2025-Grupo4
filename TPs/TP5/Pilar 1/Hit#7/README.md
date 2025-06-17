# Hit#7

Modifique el programa anterior para que reciba dos parámetros nuevos, ambos serán números y su programa debe buscar posibles soluciones solo dentro de ese rango, si en ese rango no hay soluciones debe informar que no encontró nada.

Resultado:

Para poder probar la fuerza bruta se puede hacer:

nvcc GPU/fuerzaBruta.cu -o f
./f <hash previo> <cadena> <numero minimo> <numero maximo>


Para probarlo en python seria:

py CPU/fuerzaBruta.py
