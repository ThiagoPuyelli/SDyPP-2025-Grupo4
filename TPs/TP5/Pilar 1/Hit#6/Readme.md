Hit #6 - Longitudes de prefijo en CUDA HASH

Realice mediciones sobre el programa anterior probando diferentes longitudes de prefijo. ¿Cuál es el prefijo más largo que logró encontrar? ¿Cuánto tardo? ¿Cuál es la relación entre la longitud del prefijo a buscar y el tiempo requerido para encontrarlo?

ara poder probar el minero de CUDA se puede hacer:

nvcc fuerzaBruta.cu -o f
./f <hash previo> <cadena> <rango maximo de numeros>


Para probarlo en python seria:

py CPU/fuerzaBruta.py

Ejecutando el minero de GPU con cadena 'cadeeeena' cambiando el prefijo obtenemos estos resultados:
_ Prefijo: "0", nonce: 1634, segundos: 0.082419 s, hash: 0184faf6c96e763fbc7a54f937ddeafd
_ Prefijo: "00", nonce: 568 , segundos:  0.101599 s, hash: 00e6d0620b313f6e2f1f813ee9573527
_ Prefijo: "000", nonce: 4431, segundos: 0.100366 s, hash: 000ca5d9a87cd867b5d54fb8c2e67969
_ Prefijo: "0000", nonce: 33562394, segundos: 0.097311 s, hash: 00002886cc6cccdd1f99faa5208aba9b 
_ Prefijo: "00000", nonce: 318790597, segundos: 0.110245 s, hash: 00000a8e341380341949d365c4a7ddd2
_ Prefijo: "000000", nonce: 856366564, segundos: 0.291818 s, hash: 000000fc42c8db4cd8c17e4a32eabf66 
_ Prefijo: "0000000", nonce: 638787528, segundos: 0.361616 s, hash: 0000000246a446c74fdc191d968f83bf 

En cambio con el minero de CPU con cadena 'cadeeeena' se obtuvieron los siguientes resultados:
_ Prefijo: "0", nonce: 21, segundos: 0.000096 s, hash: 0cc1c58cd762188147740f9b12a2e8ba
_ Prefijo: "00", nonce: 568, segundos: 0.000814 s, hash: 00e6d0620b313f6e2f1f813ee9573527
_ Prefijo: "000", nonce: 1713 , segundos: 0.002252 s, hash: 0006ad765fe92f87a73035cdcf82595e 
_ Prefijo: "0000", nonce: 24144, segundos: 0.028589 s, hash: 00001f1f2c06216f68a8befaa33323f7 
_ Prefijo: "00000", nonce: 216868, segundos: 0.257502 s, hash: 0000048f1ac112c4129a5c1c3946fb63  
_ Prefijo: "000000", nonce: 11848868, segundos: 11.365913 s, hash: 00000092ec5b456f76e348430b5a8af6
_ Prefijo: "0000000", nonce: 176748948, segundos: 175.687491 s, hash: 0000000ed5caef880e49e9b7f3a8eee3

En este ejemplo se puede visualizar claramente que CUDA al hacer uso de la GPU, consigue el resultado muchísimo mas rapido que CPU debido al paralelismo que presenta el hardware.