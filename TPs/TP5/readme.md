## TODO:
Pasar estado a redis para soportar multiples servidores?
como manejo el tema de selecting winner? es uno solo? agrego en redis el id del que va a calcular el ganador para cierto ciclo desde el mismo coordinador, y todos verifican que sean ese, cuando termina de calcular se borra, debe ademas tener un timestamp para que no quede bloqueado por siempre si muere al calcular el ganador


### posible mejora:
    - implementar varias transacciones por bloque (tal vez con un numero maximo)

```
docker build -t coordinador .
docker run -p 8000:8000 coordinador
```

## complejidad del desafio:
los umbrales van en el genesis

ej
90% subo 1 cero
10% bajo 1 cero


### sincronizar siempre con el mismo ntp (a nivel de sist operativo poner todos los nodos en el mismo servidor ntp)

### validar que la transaccion es valida (que realmente esa wallet tiene la plata) PONER EN EL INFORME QUE SE HACE DESPUES

### mas cosas para el bloque genesis: la pk y la privk de la cuenta base


# Informe

## Estados posibles del coordinador
- **UNSET**: el coordinador esta iniciandose, solo acepta nuevas transacciones (`Instantaneo`)
- **GIVING_TASKS**: es el tiempo de minado, el coordinador espera que los nodos le pidan trabajo y minen
- **OPEN_TO_RESULTS**: el coordinador espera los resultados de los mineros
- **SELECTING_WINNER**: el coordinador selecciona el ganador de la blockchain (`Instantaneo`)

## Recuperacion ante fallas en el coordinador
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
