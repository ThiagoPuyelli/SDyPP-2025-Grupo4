# Hit#4

Consigna:

El cálculo de funciones de hashing es ampliamente utilizado en criptografía, existen múltiples algoritmos, algunos como md5 (1991) ya son considerados inseguros y otros como sha256 (2001-2002) aún resisten la evolución y los tiempos actuales. Estos algoritmos suelen ser calculados en GPU ya que una de sus características es que son “costosos” de calcular computacionalmente.
En este punto, usted deberá escribir un programa que reciba un string por parámetro y calcule, utilizando la GPU, un md5 y devuelve el hash calculado por consola.
Puede usar librerías disponibles para este fin. Las encontrará preguntando por CUDA MD5.

Respuesta:

Para poder probar el hash md5 en GPU se puede probar utilizando:

nvcc generarHash.cu -o g
./g <texto a aplicar hash>
