# posible mejora:
    - implementar varias transacciones por bloque (tal vez con un numero maximo)

```
docker build -t coordinador .
docker run -p 8000:8000 coordinador
```

# complejidad del desafio:
los umbrales van en el genesis

ej
90% subo 1 cero
10% bajo 1 cero


# sincronizar siempre con el mismo ntp (a nivel de sist operativo poner todos los nodos en el mismo servidor ntp)

# validar que la transaccion es valida (que realmente esa wallet tiene la plata)