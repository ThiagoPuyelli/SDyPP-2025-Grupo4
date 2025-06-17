# Hit#5

Modifique el programa anterior para que reciba dos parámetros (un hash y una cadena). Ahora debe encontrar un número tal que, al concatenarlo con la cadena y calcular el hash, el resultado comience con una cadena específica proporcionada como segundo parámetro. 
Como no hay forma de adivinar cuál es ese número, deberá utilizar la GPU para probar miles o millones de combinaciones por segundo aleatoriamente hasta encontrar la correcta.
Como salida, debe mostrar el hash resultante y el número que utilizó para generarlo.

Resultado:

Para poder probar la fuerza bruta se puede hacer:

nvcc fuerzaBruta.cu -o f
./f <hash previo> <cadena> <rango maximo de numeros>


Para probarlo en python seria:

py CPU/fuerzaBruta.py
