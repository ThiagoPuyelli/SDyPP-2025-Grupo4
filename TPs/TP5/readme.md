# Para correr local dockerizado:
Nos paramos en la carpeta "Pilar2" y ejecutamos:
```shell
docker network create mining-net 

docker-compose --env-file .env up -d

docker build -t coordinadorimg ./coordinador
docker run --name coordinator --network mining-net -p 8000:8000 --env-file .env coordinadorimg

docker build -t mineroimg ./mineros/mineroCPU
docker run --name miner --network mining-net mineroimg
```


# Informe

## Modulos
- Coordinador
- Servicio de colas (Rabbitmq)
- Base de datos (Redis)
- Mineros (CPU, GPU)

## Coordinador
Coordinador de la blockchain: orquestra las transacciones que deben ser minadas, los mineros, valida, lleva registro de la blockchain, etc.

### Estados posibles del coordinador
- **UNSET**: el coordinador esta iniciandose, solo acepta nuevas transacciones (`Instantaneo`)
- **GIVING_TASKS**: es el tiempo de minado, el coordinador espera que los nodos le pidan trabajo y minen
- **OPEN_TO_RESULTS**: el coordinador espera los resultados de los mineros
- **SELECTING_WINNER**: el coordinador selecciona el ganador de la blockchain (`Instantaneo`)

### Recuperacion ante fallas en el coordinador
Cuando el servidor inicia:
- En tiempo de GIVING_TASKS:
    - Y tiene cosas en RECEIVED_CHAINS:
    `Debe calcular el ganador`
        > En este caso, entendemos o bien que el servidor se cayo recolectando resultados de los mineros y no llego a calcular el ganador del ciclo o lo hizo en el primer paso de asignar el ganador, para que esto funcione, lo primero que hacemos cuando calculamos los resultados es borrar RECEIVED_CHAINS
    - Y RECEIVED_CHAINS esta vacio:
    `Sigue el flujo normal`
        > Entendemos este como el caso en el que se cayo y levanto nuevamente dentro del periodo de GIVING_TASKS. Tambien esta la posibilidad de que se cayera cuando calcula los resultados, en el peor de los casos, se perderia otro ciclo completo entregando a los mineros transacciones invalidas (ya minadas) o las mismas transacciones porque no llego a integrarlas a la blockchain
    
- En tiempo de OPEN_TO_RESULTS:
`Sigue el flujo normalmente`
    > Tanto si el servidor cae en GIVING_TASKS como en OPEN_TO_RESULTS, si el coordinador esta en un momento que deberia dejar a los mineros que le entreguen resultados, no interrumpimos ese proceso, si hay cosas en RECEIVED_CHAINS podrian ser de este mismo ciclo o de uno anterior, en cualquier caso son validas porque no se verificaron antes como resultados

## Rabbitmq
Para encolar:
- **Transacciones Pendientes**

## Redis
Para almacenar:
- **Transacciones Activas**
- **La BlockChain**
- **Resultados de mineria**

## Mineros
Encargados de minar los bloques que les da el coordinador, pueden trabajar con GPU o CPU para minar
- Sincronizan su reloj con el del coordinador y le piden trabajo o le envian resultados segun el estado en el que este se encuentra

## Configuracion al iniciar:
- **Coordinador**: Cuando levanta, toma la configuracion del bloque genesis de la blockchain, si no hay bloque genesis, usa la configuracion que tiene por codigo y se la establece al bloque genesis `TO DO`
- **Mineros**: Cuando levantan, toman siempre la configuracion del bloque genesis de la blockchain, si no la pueden consultar, entonces no hay blockchain, y no se mina

## Posibles mejoras para una implementacion real
Si quisieramos tener un sistema de blockchain real, y acercandonos a los que hacen los sistemas mas conocidos, deberiamos implementar la posibilidad de tener varias transacciones por bloque, de manera que un minero las mine todas juntas, con un solo hash
- **Ventajas**:
    - Computacionalmente mas barato
    - Al requerir menos calculo en total, podemos dar a los mineros desafios mas altos para resolver

- **Desventajas**:
    - Si en cada ciclo tratamos de meter todas las transacciones en un solo bloque, es mas complicado ajustar el desafio, podemos terminar dando a los mineros desafios muy faciles, o muy dificiles ya que es dificil de medir el ajuste de la dificultad, porque el feedback que recibira el coordinador es binario (fue minado / no fue minado), en vez de un porcentaje de las transacciones que se llegaron a minar

Podriamos tener soluciones intermedias, como un maximo de transacciones por bloque, pero todo depende de lo que queramos implementar, no es lo mismo un sistema de criptomonedas, que una blockchain para almacenar logs.

## Mejoras posibles para futuras iteraciones

### Validar la transaccion:
- Que la firma de la transaccion este correcta
- Que la cuenta tenga saldo suficiente (verificar la blockchain + las transacciones que fueron solicitadas pero no minadas todavia)

### Recompensar a los mineros

### Sincronizar siempre con el mismo ntp (a nivel de sist. operativo poner todos los nodos en el mismo servidor ntp)
- En GCP de esto se encarga el servicio de cloud, se configuran todos contra Google

### Agregar una transaccion pseudo-valida al bloque genesis para poder empezar a hacer otras transacciones a partir de esa

### Acomodar dinamicamente la complejidad del desafio:
- Los umbrales van para definirlo van en el bloque genesis (ej.)
- Si el 90% de las transacciones o mas fueron minadas agrego un caracter al prefijo
- Si menos del 10% fueron minadas bajo le saco un caracter al prefijo

### Pool de mineria
- Implementar un pool, que reparta trabajo a varios mineros y se comporte como uno frente al coordinador


# Respuestas e informe PILAR 1

## Hit #1 

Para poder ejecutar código CUDA, no contamos con el hardware necesario, por lo tanto vamos a configurar un manifiesto de kubernetes con el fin de a partir de un pod en el cluster de Alejandro podamos pasar el código por bash, luego compilar y ejecutarlo desde ahí.

## Hit #2 

Para poder ejecutar el código en el cluster entonces definimos este manifiesto:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cuda-dev
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cuda-dev
  template:
    metadata:
      labels:
        app: cuda-dev
    spec:
      runtimeClassName: nvidia
      containers:
      - name: cuda-container
        image: nvidia/cuda:12.4.1-devel-ubuntu22.04
        command: ["/bin/bash", "-c", "sleep infinity"]
        resources:
          limits:
            nvidia.com/gpu: 1
      restartPolicy: Always
```

Al aplicar este manifiesto en nuestro namespace llamado grupo4, entramos desde la consola del pod inicializado y desde ahí escribimos el código a compilar y ejecutar.

## Hit #3

El repositorio de CCCL (CUDA Core Compute Libraries) trata sobre qué CCCL es un conjunto de bibliotecas desarrolladas por NVIDIA para facilitar la programación paralela en GPU utilizando CUDA. Estas bibliotecas ofrecen diferentes niveles de abstracción, desde interfaces de alto nivel hasta operaciones de bajo nivel optimizadas para GPU. Esta idea surgió a través de tres librerías que facilitaron mucho el desarrollo de la programación paralela en placas de video nvidia el cual se tratan de thrust, CUB y libcudacxx. En si éstas librerías tienen sus propias versiones, el cual lo que hace CCCL es de forma semántica definir una versión de cada uno a manejar.

**Thrust**: Ésta librería trata de una abstracción de alto nivel que permite de una forma sencilla hacer uso de la GPU de una forma muy productiva sin necesidad de tener que definir uno cuando usarla, adjuntamos un ejemplo de código:

```c
#include <thrust/sort.h>
#include <thrust/device_vector.h>

int main() {
    thrust::device_vector<int> vec = {5, 3, 1, 4, 2};
    thrust::sort(vec.begin(), vec.end()); // Ordena en la GPU
    return 0;
}
```

En este caso no hacemos el pasaje a GPU, pero la librería de forma abstracta si que lo hace.
Para ejecutar el código de ejemplo de la página de nvidia sobre thrust, no hizo falta instalar nada, con simplemente utilizar nvcc ya fue posible su compilación y ejecución, este es el código:

```c
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>

int main() {
// Generate 32M random numbers serially.
thrust::default_random_engine rng(1337);
thrust::uniform_int_distribution<int> dist;
thrust::host_vector<int> h_vec(32 << 20);
thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });

// Transfer data to the device.
thrust::device_vector<int> d_vec = h_vec;

// Sort data on the device.
thrust::sort(d_vec.begin(), d_vec.end());

// Transfer data back to host.
thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
}
```

Lo que realiza es generar una serie de números random y luego los ordena haciendo uso de la GPU. En sí viendo este código y la guia para empezar a programar con el uso de thrust concluimos que es una muy buena herramienta para poder programar código CUDA y hacer uso de la GPU sin necesidad de realmente meterse tanto en el código de definir quien es el que ejecuta realmente las instrucciones, como tampoco tenemos que pensar como es que esto se ejecuta de forma paralela.

Por esto en thrust existe el concepto de static dispatch y dynamic dispatch, que justamente trata de este concepto de definir en donde se alojan las variables y quien ejecuta la instrucción, el cual el static trata de que lo define en tiempo de compilación cosa que es más eficiente, en cambio el dynamic trata de que tiene que verlo en tiempo de ejecución. La ayuda que trae esta librería y muchas más al respecto de esto, sirven ya que realmente vienen con operaciones muy eficientes para lo que es la ejecución de código en placa de video, y de esta manera nosotros lo hacemos de forma abstracta.
