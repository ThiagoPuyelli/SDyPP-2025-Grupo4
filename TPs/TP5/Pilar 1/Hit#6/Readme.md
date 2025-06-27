Hit #6 - Longitudes de prefijo en CUDA HASH

Realice mediciones sobre el programa anterior probando diferentes longitudes de prefijo. ¿Cuál es el prefijo más largo que logró encontrar? ¿Cuánto tardo? ¿Cuál es la relación entre la longitud del prefijo a buscar y el tiempo requerido para encontrarlo?

ara poder probar el minero de CUDA se puede hacer:

nvcc fuerzaBruta.cu -o f
./f <hash previo> <cadena> <rango maximo de numeros>


Para probarlo en python seria:

py CPU/fuerzaBruta.py

Ejecutando el minero de CPU con cadena 'holaquetal' cambiando el prefijo obtenemos estos resultados:
_ Prefijo: "0", nonce/s: 0.000031 s.
_ Prefijo: "00", nonce/s:  0.000252 s.
_ Prefijo: "000", nonce/s: 0.006380 s.
_ Prefijo: "0000", nonce/s: 0.015636 s.
_ Prefijo: "00000", nonce/s: 0.488750 s.
_ Prefijo: "000000", nonce/s: 7.958972 s.
_ Prefijo: "0000000", nonce/s: 254.35034 s.

En cambio con el minero de GPU con cadena 'holaquetal' se obtuvieron los siguientes resultados:
_ Prefijo: "0", Tiempo de ejecucion 0.094610 s, nonce: 2581.
_ Prefijo: "00", Tiempo de ejecucion 0.18 s, nonce: 0,0025675675675676 s.
_ Prefijo: "000", Tiempo de ejecucion 0.18 s, nonce: 0,2825 s.
_ Prefijo: "0000", Tiempo de ejecucion 0.18 s, nonce: 0.8 s.
_ Prefijo: "00000", Tiempo de ejecucion 1.35 s, nonce: 1.35 s,.
_ Prefijo: "000000", Tiempo de ejecucion 1.556371 s, nonce: 1248279992.

En este ejemplo se puede visualizar claramente que CUDA al hacer uso de la GPU, consigue el resultado muchísimo mas rapido que CPU debido al paralelismo que presenta el hardware.